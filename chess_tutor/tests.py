"""API tests use deterministic legal engine doubles; real engines have opt-in tests."""
import io
import json
from contextlib import contextmanager
from unittest.mock import patch

import chess
import chess.pgn
from django.test import Client, TestCase

from .models import PlayerProfile
from .opponent import EngineUnavailable
from .views import apply_move, initial_state


class TestRoom:
    def __init__(self):
        self.fail_reply = False

    @contextmanager
    def turn(self):
        yield self

    def status(self):
        return {'ready': True, 'engine': 'Stockfish test double', 'model': 'Maia test double', 'engine_error': None}

    def observe(self, board, move, policy):
        prior = {m.uci(): 1 / board.legal_moves.count() for m in board.legal_moves}
        data = policy.learn(board, prior, move.uci())
        return {**data, 'loss_cp': 0, 'raw_loss_cp': 0, 'alternative': board.san(move),
                'engine': 'Stockfish test double', 'predicted_move': board.san(chess.Move.from_uci(data['top_prediction'])),
                'prior_prediction': board.san(chess.Move.from_uci(max(prior, key=prior.get))),
                'quality': 'Sound', 'fen_before': board.fen(), 'move_uci': move.uci(),
                'prior_hit': max(prior, key=prior.get) == move.uci(), 'personal_hit': data['top_prediction'] == move.uci()}

    def choose(self, board, policy, mode):
        if self.fail_reply:
            raise EngineUnavailable('The real engine is unavailable; no fallback.')
        move = sorted(board.legal_moves, key=lambda m: m.uci())[0]
        san = board.san(move)
        return move, {'move': san, 'baseline_move': san, 'prior_move': san,
                      'personal_changed': False, 'engine_changed': False, 'engine_cost_cp': 0,
                      'expected_regret_cp': 0, 'prior_expected_regret_cp': 0,
                      'candidates': [], 'replies': [], 'depth': 1, 'nodes': 1}


class GameTests(TestCase):
    def setUp(self):
        self.room = TestRoom()
        self.patch_room = patch('chess_tutor.views.engine_room', return_value=self.room)
        self.patch_room.start()
        self.addCleanup(self.patch_room.stop)
        runtime = patch('chess_tutor.views.current_maia_fingerprint', return_value='test-prior')
        runtime.start(); self.addCleanup(runtime.stop)
        self.client.get('/')

    def state(self, client=None):
        return (client or self.client).get('/api/state/').json()

    def action(self, action, **extra):
        return self.client.post('/api/action/', data=json.dumps(
            {'action': action, 'revision': self.state()['revision'], **extra}), content_type='application/json')

    def test_play_persistence_new_game_and_exports(self):
        response = self.action('move', move='e4')
        self.assertEqual(response.status_code, 200, response.content)
        state = response.json()
        self.assertEqual(len(state['moves']), 2)
        self.assertEqual(state['profile']['samples'], 1)
        self.assertEqual(state['profile']['prior_log_loss'], state['profile']['personal_log_loss'])
        self.assertEqual(state['turn'], 'white')
        self.assertEqual(self.state()['fen'], state['fen'])
        restored = Client(); restored.cookies = self.client.cookies
        self.assertEqual(self.state(restored)['profile']['samples'], 1)
        game = chess.pgn.read_game(io.StringIO(self.client.get('/api/export/').content.decode()))
        self.assertEqual(game.end().board().fen(), state['fen'])
        data = self.client.get('/api/export/?format=json').json()
        self.assertEqual(data['version'], 2)
        self.assertEqual(data['state']['policy']['samples'], 1)
        self.assertEqual(data['state']['events'][0]['samples_before'], 0)
        self.assertEqual(data['state']['events'][0]['fen_before'], chess.STARTING_FEN)
        reset = self.action('new_game', mode='baseline').json()
        self.assertEqual(reset['profile']['samples'], 1)
        self.assertEqual(reset['moves'], [])
        self.assertIsNone(reset['decision'])
        self.assertEqual(reset['mode'], 'baseline')
        self.assertEqual(len(reset['archive']), 1)

    def test_separate_browser_has_separate_personal_policy(self):
        self.action('move', move='e4')
        other = Client()
        self.assertEqual(self.state(other)['profile']['samples'], 0)
        self.assertEqual(self.state(other)['moves'], [])

    def test_bad_move_and_stale_request_do_not_learn(self):
        before = self.state()
        self.assertEqual(self.action('move', move='e5').status_code, 400)
        self.assertEqual(self.state(), before)
        self.action('move', move='e4')
        after = self.state()
        stale = self.client.post('/api/action/', data=json.dumps(
            {'action': 'move', 'move': 'Nf3', 'revision': before['revision']}), content_type='application/json')
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(self.state(), after)

    def test_engine_failure_rolls_back_human_move_and_learning(self):
        before = self.state()
        self.room.fail_reply = True
        response = self.action('move', move='e4')
        self.assertEqual(response.status_code, 503)
        self.assertIn('no fallback', response.json()['error'])
        self.assertEqual(self.state(), before)

    def test_terminal_move_and_underpromotion(self):
        profile = PlayerProfile.objects.get(pk=self.client.session['nemesis_profile'])
        profile.state = initial_state()
        profile.state['moves'] = ['e2e4','e7e5','d1h5','b8c6','f1c4','g8f6']
        profile.save()
        self.room.fail_reply = True
        result = self.action('move', move='Qxf7#').json()
        self.assertEqual(result['result'], '1-0')
        self.assertEqual(len(result['moves']), 7)
        self.assertEqual(result['profile']['samples'], 1)
        board = chess.Board('7k/P7/8/8/8/8/8/7K w - - 0 1')
        state = initial_state()
        with patch('chess_tutor.views.restore_board', return_value=board):
            apply_move(state, 'a7a8n')
        self.assertEqual(state['result'], '1/2-1/2')
        self.assertEqual(state['moves'], ['a7a8n'])

    def test_resign_draw_claim_and_explicit_reset(self):
        self.assertEqual(self.action('claim_draw').status_code, 400)
        self.action('move', move='e4')
        resigned = self.action('resign').json()
        self.assertEqual(resigned['result'], '0-1')
        self.assertTrue(resigned['game_over'])
        self.assertEqual(self.action('move', move='Nf3').status_code, 400)
        reset = self.action('forget').json()
        self.assertEqual(reset['profile']['samples'], 0)
        self.assertEqual(reset['moves'], [])
        self.assertIsNone(reset['profile']['prior_log_loss'])

    def test_old_regret_model_is_not_interpreted_as_choice_model(self):
        profile = PlayerProfile.objects.get(pk=self.client.session['nemesis_profile'])
        old = {'moves': ['e2e4','e7e5'], 'games': 3, 'completed': 1, 'mode': 'adaptive',
               'archive': [], 'result': '*', 'network': {'samples': 40}, 'stats': {'Development': 12}}
        profile.state = old; profile.save()
        migrated = self.state()
        self.assertEqual(migrated['moves'], ['e4','e5'])
        self.assertEqual(migrated['profile']['samples'], 0)
        profile.refresh_from_db()
        self.assertEqual(profile.state['legacy_v1'], old)
        self.assertEqual(profile.state['version'], 2)

    def test_csrf_methods_and_bad_json(self):
        client = Client(enforce_csrf_checks=True); client.get('/')
        payload = json.dumps({'action': 'move', 'revision': 0, 'move': 'e4'})
        self.assertEqual(client.post('/api/action/', data=payload, content_type='application/json').status_code, 403)
        for bad in ('[]', 'null', '{', '{"revision":false}', '{"revision":0,"action":"nope"}'):
            self.assertEqual(self.client.post('/api/action/', data=bad, content_type='application/json').status_code, 400)
        self.assertEqual(self.client.get('/api/action/').status_code, 405)
