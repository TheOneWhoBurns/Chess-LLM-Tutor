"""Practice correctness, persistence, grading, and separation from live learning."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
import io
import json
import os
from unittest import skipUnless
from unittest.mock import Mock, patch

import chess
from django.core.management import call_command
from django.test import Client, TestCase
from django.utils import timezone

from .models import PlayerProfile, PracticeAttempt, PracticeEvaluation, PracticeLesson
from .opponent import EngineRoom, EngineUnavailable, MATE
from .personal_profile import named_profile_id
from .practice import CONFIRM_NODES, acceptable, create_lesson, extract_saved_mistakes
from .tests import TestRoom
from .views import initial_state


class PracticeRoom:
    def __init__(self):
        self.stockfish = Mock(name='engine')
        self.stockfish.name = 'Stockfish test'
        self.stockfish.rank.side_effect = self.rank

    @contextmanager
    def turn(self):
        yield self

    def close(self):
        pass

    def rank(self, board, **kwargs):
        scores = {'e2e4': 120, 'd2d4': 100, 'a2a3': -200,
                  'e7e5': 120, 'd7d5': 100, 'a7a6': -200}
        rows = []
        for move in board.legal_moves:
            after = board.copy()
            after.push(move)
            pv = [move.uci()]
            if not after.is_game_over():
                pv.append(next(iter(after.legal_moves)).uci())
            rows.append({'move': move, 'score': scores.get(move.uci(), 0), 'mate': None,
                         'depth': 18, 'nodes': kwargs['nodes'], 'pv': pv})
        return sorted(rows, key=lambda row: -row['score'])


class PracticeTests(TestCase):
    def setUp(self):
        for target, value in [('chess_tutor.views.active_username', None),
                              ('chess_tutor.views.engine_room', TestRoom())]:
            p = patch(target, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        self.client.get('/practice/')
        self.profile = PlayerProfile.objects.get(pk=self.client.session['nemesis_profile'])
        self.room = PracticeRoom()
        self.lesson = create_lesson(self.profile, chess.Board(), chess.Move.from_uci('a2a3'),
                                    'Local game 1 · move 1', known_loss=300, room=self.room)

    def post(self, action, *, client=None, **extra):
        return (client or self.client).post('/api/practice/',
            json.dumps({'action': action, **extra}), content_type='application/json')

    def begin(self):
        response = self.post('start')
        self.assertEqual(response.status_code, 200)
        return response.json()['attempt']

    def answer(self, attempt, move='e4', **extra):
        return self.post('submit', id=attempt['id'], revision=attempt['revision'], move=move, **extra)

    def make_due(self):
        self.lesson.due_at = timezone.now() - timedelta(seconds=1)
        self.lesson.save(update_fields=['due_at'])

    def test_confirmation_preserves_history_and_accepts_sound_alternatives(self):
        call = self.room.stockfish.rank.call_args
        self.assertEqual(call.kwargs['nodes'], CONFIRM_NODES)
        self.assertEqual(call.kwargs['count'], 20)
        self.assertEqual(set(self.lesson.evidence['accepted']), {'e2e4', 'd2d4'})
        response = self.answer(self.begin(), 'd4')
        self.assertTrue(response.json()['attempt']['result']['unaided'])
        self.assertEqual(response.json()['summary']['delayed_checks'], 0)

    def test_answers_are_hidden_and_start_is_idempotent(self):
        attempt = self.begin()
        self.assertNotIn('result', attempt)
        self.assertNotIn('accepted', attempt)
        self.assertNotIn('best', attempt)
        self.assertNotIn('original', attempt)
        self.assertEqual(attempt['hints'], [])
        self.assertEqual(self.begin()['id'], attempt['id'])
        self.assertEqual(PracticeAttempt.objects.count(), 1)

    def test_hint_persists_and_cannot_be_removed_by_client(self):
        attempt = self.begin()
        response = self.post('hint', id=attempt['id'], revision=attempt['revision'])
        hinted = response.json()['attempt']
        self.assertEqual(len(hinted['hints']), 1)
        refreshed = self.client.get('/api/practice/').json()['attempt']
        self.assertEqual(refreshed, hinted)
        result = self.answer(hinted, hint_level=0).json()['attempt']['result']
        self.assertTrue(result['correct'])
        self.assertFalse(result['unaided'])
        self.lesson.refresh_from_db()
        self.assertEqual(self.lesson.stage, 0)

    def test_stale_tab_cannot_grade_as_unaided_after_hint(self):
        attempt = self.begin()
        self.post('hint', id=attempt['id'], revision=attempt['revision'])
        response = self.answer(attempt)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(len(response.json()['attempt']['hints']), 1)
        self.assertEqual(PracticeAttempt.objects.filter(finished_at__isnull=False).count(), 0)

    def test_delayed_checks_require_due_date_and_advance_schedule(self):
        result = self.answer(self.begin()).json()
        self.assertEqual(result['attempt']['result']['interval_days'], 1)
        self.assertIsNone(self.post('start').json()['attempt'])
        self.make_due()
        review = self.begin()
        self.assertTrue(review['is_review'])
        result = self.answer(review).json()
        self.assertEqual(result['attempt']['result']['interval_days'], 3)
        self.assertEqual(result['summary']['delayed_checks'], 1)
        self.assertEqual(result['summary']['delayed_unaided'], 1)

    def test_failed_review_resets_interval_without_hiding_failure(self):
        self.answer(self.begin())
        self.make_due()
        result = self.answer(self.begin(), 'a3').json()
        self.assertFalse(result['attempt']['result']['correct'])
        self.assertEqual(result['attempt']['result']['interval_days'], 1)
        self.assertEqual(result['summary']['delayed_checks'], 1)
        self.assertEqual(result['summary']['delayed_unaided'], 0)

    def test_reveal_is_not_a_success_and_duplicate_submit_does_not_count_twice(self):
        attempt = self.begin()
        revealed = self.post('reveal', id=attempt['id'], revision=attempt['revision']).json()
        self.assertFalse(revealed['attempt']['result']['unaided'])
        repeated = self.answer(attempt).json()
        self.assertEqual(repeated, revealed)
        self.assertEqual(repeated['summary']['completed'], 1)
        self.assertTrue(self.client.get('/api/practice/').json()['attempt']['finished'])

    def test_invalid_move_rolls_back_attempt_revision_and_schedule(self):
        attempt = self.begin()
        due = self.lesson.due_at
        for move in ('e9', 'e2e5', '0000', '--'):
            with self.subTest(move=move):
                self.assertEqual(self.answer(attempt, move).status_code, 400)
        saved = PracticeAttempt.objects.get(pk=attempt['id'])
        self.assertEqual(saved.revision, 0)
        self.assertIsNone(saved.finished_at)
        self.lesson.refresh_from_db()
        self.assertEqual(self.lesson.due_at, due)

    def test_practice_leaves_live_game_and_learning_unchanged_and_exports_evidence(self):
        state, revision = deepcopy(self.profile.state), self.profile.revision
        result = self.answer(self.begin(), reasoning='I expect a developing move.').json()
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.state, state)
        self.assertEqual(self.profile.revision, revision)
        exported = self.client.get('/api/export/?format=json').json()['practice']
        self.assertEqual(exported[0]['attempts'][0]['result'], result['attempt']['result'])
        self.assertIn('history', exported[0]['evidence'])

    def test_other_browser_cannot_access_attempt(self):
        attempt = self.begin()
        other = Client()
        other.get('/practice/')
        self.assertIsNone(other.get('/api/practice/').json()['attempt'])
        self.assertEqual(self.post('hint', client=other, id=attempt['id'], revision=0).status_code, 404)

    def test_csrf_json_and_method_validation(self):
        client = Client(enforce_csrf_checks=True)
        client.get('/practice/')
        self.assertEqual(self.post('start', client=client).status_code, 403)
        self.assertEqual(self.client.put('/api/practice/').status_code, 405)
        self.assertEqual(self.client.post('/api/practice/', '[]', content_type='application/json').status_code, 400)
        self.assertEqual(self.post('submit', id='invalid', revision=0).status_code, 400)

    def test_reset_clears_lessons_attempts_and_evaluations(self):
        self.begin()
        response = self.client.post('/api/action/', json.dumps({'action':'forget', 'revision':0}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(PracticeLesson.objects.count(), 0)
        self.assertEqual(PracticeAttempt.objects.count(), 0)
        self.assertEqual(PracticeEvaluation.objects.count(), 0)

    def test_source_deduplication_and_black_to_move(self):
        self.assertIsNone(create_lesson(self.profile, chess.Board(), chess.Move.from_uci('a2a3'),
                                       'again', known_loss=300, room=self.room))
        board = chess.Board()
        board.push_san('e4')
        black = create_lesson(self.profile, board, chess.Move.from_uci('a7a6'),
                              'Imported black game', known_loss=300, room=self.room)
        self.assertEqual(black.evidence['history'], ['e2e4'])
        self.assertEqual(set(black.evidence['accepted']), {'e7e5', 'd7d5'})
        self.assertEqual(board.turn, chess.BLACK)

    def test_deeper_search_can_reject_mistake_and_remembers_checked_position(self):
        board = chess.Board()
        move = chess.Move.from_uci('d2d4')
        self.assertIsNone(create_lesson(self.profile, board, move, 'noise', known_loss=200, room=self.room))
        self.room.stockfish.rank.reset_mock()
        self.assertIsNone(create_lesson(self.profile, board, move, 'noise', known_loss=200, room=self.room))
        self.room.stockfish.rank.assert_not_called()

    def test_engine_failure_does_not_mark_a_position_as_checked(self):
        board = chess.Board()
        board.push_san('e4')
        before = PracticeEvaluation.objects.count()
        self.room.stockfish.rank.side_effect = EngineUnavailable('engine failed')
        with self.assertRaises(EngineUnavailable):
            create_lesson(self.profile, board, chess.Move.from_uci('a7a6'), 'failed', room=self.room)
        self.assertEqual(PracticeEvaluation.objects.count(), before)

    def test_saved_game_extraction_preserves_full_move_history(self):
        board = chess.Board()
        board.push_san('e4')
        board.push_san('e5')
        fen = board.fen()
        self.profile.state['moves'] = ['e2e4', 'e7e5', 'a2a3']
        self.profile.state['events'] = [{'fen_before':fen, 'move_uci':'a2a3', 'loss_cp':300}]
        with patch('chess_tutor.practice.engine_room', return_value=self.room):
            result = extract_saved_mistakes(self.profile)
        self.assertEqual(result['created'], 1)
        lesson = PracticeLesson.objects.get(evidence__fen=fen)
        self.assertEqual(lesson.evidence['history'], ['e2e4', 'e7e5'])
        self.assertEqual(self.room.stockfish.rank.call_args.args[0].move_stack,
                         [chess.Move.from_uci('e2e4'), chess.Move.from_uci('e7e5')])

    def test_mate_acceptance_preserves_a_forced_win(self):
        best = {'score':MATE - 2, 'mate':2}
        self.assertTrue(acceptable(best, {'score':MATE - 8, 'mate':8}))
        self.assertFalse(acceptable(best, {'score':MATE - 3, 'mate':None}))
        self.assertFalse(acceptable({'score':0, 'mate':None}, {'score':-MATE + 2, 'mate':-2}))

    def test_import_command_uses_cached_history_and_own_black_moves(self):
        username = 'practice_player'
        profile = PlayerProfile.objects.create(pk=named_profile_id(username), state=initial_state())
        imported = Mock(games=[{'url':'https://www.chess.com/game/123', 'user_color':'black',
                              'pgn':'1. e4 a6 *'}])
        with patch('chess_tutor.management.commands.extract_practice.cached_games', return_value=imported), \
                patch('chess_tutor.practice.engine_room', return_value=self.room):
            call_command('extract_practice', username=username, games=1, stdout=io.StringIO())
        self.assertEqual(profile.practice_lessons.count(), 1)
        self.assertEqual(profile.practice_lessons.first().evidence['original'], 'a7a6')


@skipUnless(os.environ.get('NEMESIS_TEST_REAL_ENGINES') == '1', 'Set NEMESIS_TEST_REAL_ENGINES=1')
class RealPracticeTests(TestCase):
    def test_real_engine_confirms_a_missed_mate_and_returns_legal_lines(self):
        room = EngineRoom()
        self.addCleanup(room.close)
        profile = PlayerProfile.objects.create(state=initial_state())
        board = chess.Board()
        for san in ('f3', 'e5', 'g4'):
            board.push_san(san)
        lesson = create_lesson(profile, board, board.parse_san('a6'), 'Missed mate', room=room)
        self.assertIsNotNone(lesson)
        self.assertIn('d8h4', lesson.evidence['accepted'])
        self.assertNotIn('a7a6', lesson.evidence['accepted'])
        for row in lesson.evidence['choices'].values():
            replay = board.copy()
            for uci in row['pv']:
                replay.push(replay.parse_uci(uci))
