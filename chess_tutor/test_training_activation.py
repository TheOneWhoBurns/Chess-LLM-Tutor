"""Exercise activation with artifacts produced by the actual history runner."""
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from django.test import TestCase

from scripts.train_history import run_training

from .chesscom_import import parse_game
from .deployment import activate_player
from .models import PlayerProfile
from .personal_profile import active_username, named_profile_id


class TrainingActivationIntegrationTests(TestCase):
    def test_real_runner_artifacts_activate_with_matching_counts_and_runtime(self):
        username = 'example_player'
        games = []
        for index, color, moves, result in (
                (1, 'white', '1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6', '1-0'),
                (2, 'black', '1. d4 d5 2. c4 e6 3. Nc3 Nf6', '0-1')):
            raw = {'url': f'https://www.chess.com/game/live/test-{index}',
                   'end_time': 1700000000 + index, 'rules': 'chess',
                   'time_class': 'rapid', 'time_control': '600', 'rated': True,
                   'white': {'username': username if color == 'white' else 'other_player', 'result': 'win'},
                   'black': {'username': username if color == 'black' else 'other_player', 'result': 'resigned'},
                   'pgn': f'[Event "Test game"]\n[Result "{result}"]\n\n{moves} {result}'}
            games.append(parse_game(raw, username))
        fingerprint = {'quantization': 'lc0-percent-midpoint-v2', 'test_runtime': 'fixture'}
        prior = lambda board: {move.uci(): 1 / board.legal_moves.count() for move in board.legal_moves}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output, config = directory / 'training', directory / 'player.json'
            with redirect_stdout(io.StringIO()):
                report = run_training(games, username, output, prior, policy_fingerprint=fingerprint)
            model = json.loads((output / 'deployment_model.json').read_text())
            self.assertTrue(model['complete_dataset'])
            self.assertEqual(report['summary']['games_total'], 2)
            self.assertEqual(report['summary']['train_positions'], 3)
            self.assertEqual(report['summary']['test_positions'], 3)
            self.assertEqual(report['summary']['total_positions'], 6)
            self.assertEqual(model['policy']['samples'], 6)
            with patch('chess_tutor.deployment.current_maia_fingerprint', return_value=fingerprint):
                activation = activate_player(username, output, config)
            self.assertTrue(activation['created'])
            self.assertEqual(activation['samples'], 6)
            self.assertEqual(active_username(config), username)
            profile = PlayerProfile.objects.get(pk=named_profile_id(username))
            self.assertEqual(profile.state['training'], report['summary'])
            self.assertEqual(profile.state['policy_fingerprint'], fingerprint)
            self.assertEqual(profile.state['metrics']['count'], 0)
            self.assertEqual(profile.state['moves'], [])
            original = profile.state
            with patch('chess_tutor.deployment.current_maia_fingerprint', return_value={'changed': True}):
                with self.assertRaisesRegex(ValueError, 'different Maia runtime'):
                    activate_player(username, output, config)
            profile.refresh_from_db()
            self.assertEqual(profile.state, original)
