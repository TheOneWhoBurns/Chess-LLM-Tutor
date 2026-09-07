import io
import json
from unittest.mock import MagicMock, patch

import chess
import chess.pgn
import numpy as np
from django.test import Client, SimpleTestCase, TestCase

from .models import PlayerProfile
from .nemesis import Evaluator, MATE, PlayerNetwork, choose_move, demo_candidates, features
from .views import initial_state


class LearningTests(SimpleTestCase):
    def test_uci_candidate_scores_use_root_perspective_and_close(self):
        board = chess.Board(); board.push_san('e4')
        moves = list(board.legal_moves)
        engine = MagicMock()
        engine.id = {'name': 'Test UCI'}
        engine.analyse.return_value = [
            {'pv': [move], 'score': chess.engine.PovScore(chess.engine.Cp(25 + index), chess.WHITE)}
            for index, move in enumerate(moves)]
        with patch.dict('os.environ', {'NEMESIS_UCI_ENGINE': '/test/engine'}), patch(
                'chess.engine.SimpleEngine.popen_uci', return_value=engine):
            with Evaluator() as evaluator:
                ranked = evaluator.rank(board)
                self.assertEqual(ranked[0]['score'], -25)
                self.assertEqual(ranked[0]['move'], moves[0])
                self.assertEqual(evaluator.name, 'Test UCI')
        engine.quit.assert_called_once()

    def test_incomplete_uci_output_falls_back_to_complete_demo_search(self):
        engine = MagicMock(); engine.id = {'name': 'Test UCI'}
        engine.analyse.return_value = []
        board = chess.Board()
        with patch.dict('os.environ', {'NEMESIS_UCI_ENGINE': '/test/engine'}), patch(
                'chess.engine.SimpleEngine.popen_uci', return_value=engine):
            with Evaluator() as evaluator:
                ranked = evaluator.rank(board)
                self.assertEqual(len(ranked), board.legal_moves.count())
                self.assertIn('fallback', evaluator.name)

    def test_missing_uci_binary_falls_back(self):
        with patch.dict('os.environ', {'NEMESIS_UCI_ENGINE': '/missing/engine'}), patch(
                'chess.engine.SimpleEngine.popen_uci', side_effect=FileNotFoundError), self.assertLogs(
                    'chess_tutor.nemesis', level='WARNING'):
            with Evaluator() as evaluator:
                self.assertEqual(len(evaluator.rank(chess.Board())), 20)
                self.assertIn('unavailable', evaluator.name)

    def test_network_learns_context_and_survives_serialization(self):
        network = PlayerNetwork()
        vulnerable = np.zeros(12); vulnerable[0] = 1; vulnerable[6] = 1
        comfortable = np.zeros(12); comfortable[2] = 1; comfortable[7] = 1
        before = float(network.predict(vulnerable))
        for _ in range(50):
            network.learn(vulnerable, 300)
            network.learn(comfortable, 0)
        self.assertGreater(network.predict(vulnerable), .8)
        self.assertLess(network.predict(comfortable), .2)
        self.assertGreater(network.predict(vulnerable), before)
        loaded = PlayerNetwork(json.loads(json.dumps(network.dump())))
        self.assertAlmostEqual(float(loaded.predict(vulnerable)), float(network.predict(vulnerable)))
        self.assertEqual(loaded.samples, 100)

    def test_no_personalization_before_eight_observations(self):
        net = PlayerNetwork()
        for _ in range(7):
            net.learn(features(chess.Board()), 200)
        self.assertEqual(net.influence, 0)
        net.learn(features(chess.Board()), 200)
        self.assertGreater(net.influence, 0)

    def test_trained_network_changes_a_real_chess_continuation(self):
        board = chess.Board('rnbqkbnr/pp2pppp/2pp4/8/4P3/1P3P2/P1PP2PP/RNBQKBNR b KQkq - 0 3')
        candidates = demo_candidates(board)
        board.push(candidates[0]['move']); comfortable = features(board); board.pop()
        board.push_uci('d8b6'); vulnerable = features(board); board.pop()
        network = PlayerNetwork()
        for _ in range(120):
            network.learn(comfortable, 0)
            network.learn(vulnerable, 300)
        baseline, _ = choose_move(board, candidates, network, adaptive=False)
        personal, changed = choose_move(board, candidates, network)
        self.assertEqual(baseline['move'].uci(), 'b8d7')
        self.assertEqual(personal['move'].uci(), 'd8b6')
        self.assertTrue(changed)
        self.assertLessEqual(baseline['score'] - personal['score'], 65)

    def test_features_are_finite_and_bounded(self):
        for fen in (chess.STARTING_FEN, '8/8/8/8/3k4/8/3P4/3K4 w - - 0 50'):
            for color in chess.COLORS:
                x = features(chess.Board(fen), color)
                self.assertEqual(x.shape, (12,))
                self.assertTrue(np.isfinite(x).all())
                self.assertTrue((np.abs(x) <= 1).all())

    def test_reranking_changes_move_but_respects_safety_and_baseline(self):
        board = chess.Board(); board.push_san('e4')
        candidates = [{"move": chess.Move.from_uci('e7e5'), "score": 20},
                      {"move": chess.Move.from_uci('c7c5'), "score": 10},
                      {"move": chess.Move.from_uci('f7f6'), "score": -100}]
        class Model:
            influence = 1
            def predict(self, x):
                return 0 if board.peek().uci() == 'e7e5' else 1
        move, changed = choose_move(board, candidates, Model())
        self.assertEqual(move['move'].uci(), 'c7c5')
        self.assertTrue(changed)
        move, changed = choose_move(board, candidates, Model(), adaptive=False)
        self.assertEqual(move['move'].uci(), 'e7e5')
        self.assertFalse(changed)
        candidates[0]['score'] = MATE - 5
        move, _ = choose_move(board, candidates, Model())
        self.assertEqual(move['move'].uci(), 'e7e5')
        self.assertEqual(len(board.move_stack), 1)

    def test_demo_finds_mate_and_preserves_board(self):
        board = chess.Board()
        for san in ('f3','e5','g4'): board.push_san(san)
        before = board.fen()
        ranked = demo_candidates(board)
        self.assertEqual(ranked[0]['move'].uci(), 'd8h4')
        self.assertEqual(ranked[0]['score'], MATE)
        self.assertEqual(board.fen(), before)

    def test_legacy_evaluation_uses_moving_players_perspective(self):
        from .maia_engine import MaiaEngine
        engine = object.__new__(MaiaEngine)
        engine.engine = None
        engine.BLUNDER_THRESHOLD = -200
        engine.MISTAKE_THRESHOLD = -100
        engine.GOOD_MOVE_THRESHOLD = 50
        engine.EXCELLENT_MOVE_THRESHOLD = 150
        for color, values in ((chess.WHITE, [100, -150]), (chess.BLACK, [-100, 150])):
            board = chess.Board(); board.turn = color
            with patch.object(engine, 'get_position_evaluation', side_effect=values):
                result = engine.evaluate_move_quality(board, next(iter(board.legal_moves)))
            self.assertEqual(result['evaluation_difference'], -250)
            self.assertEqual(result['quality'], 'Blunder')


class GameTests(TestCase):
    def setUp(self):
        self.client.get('/')

    def state(self, client=None):
        return (client or self.client).get('/api/state/').json()

    def action(self, action, **extra):
        return self.client.post('/api/action/', data=json.dumps(
            {'action': action, 'revision': self.state()['revision'], **extra}), content_type='application/json')

    def test_play_persist_new_game_and_export(self):
        response = self.action('move', move='e4')
        self.assertEqual(response.status_code, 200, response.content)
        state = response.json()
        self.assertEqual(len(state['moves']), 2)
        self.assertEqual(state['profile']['samples'], 1)
        self.assertEqual(state['turn'], 'white')
        self.assertEqual(self.state()['fen'], state['fen'])
        game = chess.pgn.read_game(io.StringIO(self.client.get('/api/export/').content.decode()))
        self.assertEqual(game.end().board().fen(), state['fen'])
        exported = self.client.get('/api/export/?format=json').json()
        self.assertEqual(exported['state']['network']['samples'], 1)
        reset = self.action('new_game', mode='baseline').json()
        self.assertEqual(reset['profile']['samples'], 1)
        self.assertEqual(reset['moves'], [])
        self.assertEqual(reset['mode'], 'baseline')
        self.assertEqual(len(reset['archive']), 1)

    def test_profile_survives_new_client_with_same_cookie(self):
        self.action('move', move='Nf3')
        client = Client(); client.cookies = self.client.cookies
        self.assertEqual(self.state(client)['profile']['samples'], 1)

    def test_separate_browser_does_not_share_board(self):
        self.action('move', move='e4')
        other = Client()
        self.assertEqual(self.state(other)['moves'], [])
        self.assertEqual(self.state(other)['profile']['samples'], 0)

    def test_invalid_move_and_stale_revision_cannot_train(self):
        before = self.state()
        response = self.action('move', move='e5')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.state(), before)
        self.action('move', move='e4')
        after = self.state()
        response = self.client.post('/api/action/', data=json.dumps(
            {'action':'move', 'revision':before['revision'], 'move':'Nf3'}), content_type='application/json')
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.state(), after)

    def test_engine_failure_rolls_back_move_and_model(self):
        before = self.state()
        from .nemesis import Evaluator
        original = Evaluator.rank
        def fail_on_reply(self, board):
            if board.turn == chess.BLACK: raise RuntimeError('simulated engine fault')
            return original(self, board)
        with patch.object(Evaluator, 'rank', fail_on_reply), self.assertLogs('chess_tutor.views', level='ERROR'):
            response = self.action('move', move='e4')
        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.state(), before)

    def test_resignation_draw_claim_and_forget(self):
        self.assertEqual(self.action('claim_draw').status_code, 400)
        self.action('move', move='e4')
        resigned = self.action('resign').json()
        self.assertTrue(resigned['game_over'])
        self.assertEqual(resigned['result'], '0-1')
        self.assertEqual(self.action('move', move='Nf3').status_code, 400)
        self.assertEqual(self.action('resign').status_code, 400)
        forgotten = self.action('forget').json()
        self.assertEqual(forgotten['profile']['samples'], 0)
        self.assertEqual(forgotten['profile']['completed'], 0)
        self.assertEqual(forgotten['moves'], [])

    def test_checkmate_from_player_does_not_request_reply(self):
        profile = PlayerProfile.objects.get(pk=self.client.session['nemesis_profile'])
        profile.state = initial_state()
        profile.state['moves'] = ['e2e4','e7e5','d1h5','b8c6','f1c4','g8f6']
        profile.save()
        result = self.action('move', move='Qxf7#').json()
        self.assertEqual(result['result'], '1-0')
        self.assertEqual(len(result['moves']), 7)
        self.assertTrue(result['game_over'])
        self.assertEqual(result['profile']['samples'], 1)

    def test_underpromotion_and_castling(self):
        profile = PlayerProfile.objects.get(pk=self.client.session['nemesis_profile'])
        state = initial_state()
        state['moves'] = ['e2e4','e7e5','g1f3','b8c6','f1c4','f8c5']
        profile.state = state; profile.save()
        result = self.action('move', move='O-O')
        self.assertEqual(result.status_code, 200, result.content)
        self.assertEqual(result.json()['moves'][6], 'O-O')
        board = chess.Board('7k/P7/8/8/8/8/8/7K w - - 0 1')
        from .views import apply_move
        promotion_state = initial_state()
        # Exercise promotion and terminal handling directly from a constructed position.
        with patch('chess_tutor.views.restore_board', return_value=board):
            apply_move(promotion_state, 'a7a8n')
        self.assertEqual(promotion_state['moves'], ['a7a8n'])
        self.assertEqual(promotion_state['result'], '1/2-1/2')
        self.assertEqual(promotion_state['network']['samples'], 1)

    def test_csrf_and_malformed_requests(self):
        client = Client(enforce_csrf_checks=True); client.get('/')
        payload = json.dumps({'action':'move','revision':0,'move':'e4'})
        response = client.post('/api/action/', data=payload, content_type='application/json')
        self.assertEqual(response.status_code, 403)
        for bad in ('[]', 'null', '{', '{"revision":false}', '{"revision":0,"action":"nonsense"}'):
            self.assertEqual(self.client.post('/api/action/', data=bad, content_type='application/json').status_code, 400)
        self.assertEqual(self.client.get('/api/action/').status_code, 405)
