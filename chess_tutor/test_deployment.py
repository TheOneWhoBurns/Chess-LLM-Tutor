"""Completed history activation preserves profiles and keeps live metrics separate."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import chess
from django.test import Client, TestCase

from .deployment import activate_player
from .models import PlayerProfile
from .personal_profile import active_username, named_profile_id
from .player_policy import PersonalPolicy
from .tests import TestRoom
from .views import initial_state


class DeploymentTests(TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = self.directory / "player.json"
        policy = PersonalPolicy()
        board = chess.Board()
        prior = {m.uci(): 1 / 20 for m in board.legal_moves}
        for _ in range(2):
            policy.learn(board, prior, "e2e4")
        identity = {"username": "example_player", "dataset_fingerprint": "test-dataset",
                    "policy_fingerprint": "test-prior"}
        self.model = {**identity, "complete_dataset": True, "policy": policy.dump(),
                      "purpose": "all_selected_games_for_live_play"}
        self.report = {**identity, "training_complete": True, "summary": {"username": "example_player", "games_total": 2,
            "train_games": 1, "test_games": 1, "train_positions": 1, "test_positions": 1,
            "total_positions": 2, "prior_log_loss": 2., "personal_log_loss": 1.9,
            "prior_accuracy": 0., "personal_accuracy": 1., "trained_at": "2026-01-01T00:00:00Z"}}
        self.write_artifacts()
        engine = patch('chess_tutor.views.engine_room', return_value=TestRoom())
        engine.start(); self.addCleanup(engine.stop)
        for target in ('chess_tutor.deployment.current_maia_fingerprint', 'chess_tutor.views.current_maia_fingerprint'):
            runtime = patch(target, return_value='test-prior')
            runtime.start(); self.addCleanup(runtime.stop)

    def write_artifacts(self):
        (self.directory / "deployment_model.json").write_text(json.dumps(self.model))
        (self.directory / "report.json").write_text(json.dumps(self.report))

    def activate(self):
        return activate_player("Example_Player", self.directory, self.config)

    def test_activation_shares_profile_and_does_not_erase_old_browser_data(self):
        old = initial_state(); old['moves'] = ['d2d4', 'd7d5']
        previous = PlayerProfile.objects.create(state=old)
        self.assertTrue(self.activate()['created'])
        self.assertEqual(active_username(self.config), 'example_player')
        with patch('chess_tutor.views.active_username', return_value='example_player'):
            first, second = Client(), Client()
            state = first.get('/api/state/').json()
            self.assertEqual(state['profile']['samples'], 2)
            self.assertEqual(state['profile']['live_samples'], 0)
            self.assertIsNone(state['profile']['personal_log_loss'])
            self.assertEqual(state['profile']['training'], self.report['summary'])
            response = first.post('/api/action/', data=json.dumps(
                {'action': 'move', 'move': 'e4', 'revision': state['revision']}),
                content_type='application/json')
            self.assertEqual(response.status_code, 200, response.content)
            updated = second.get('/api/state/').json()
            self.assertEqual(updated['profile']['samples'], 3)
            self.assertEqual(updated['profile']['live_samples'], 1)
            profile = PlayerProfile.objects.get(pk=named_profile_id('example_player'))
            self.assertEqual(updated['profile']['personal_log_loss'], profile.state['metrics']['personal_loss'])
            self.assertEqual(updated['moves'], response.json()['moves'])
            self.assertEqual(first.session['nemesis_profile'], second.session['nemesis_profile'])
        previous.refresh_from_db()
        self.assertEqual(previous.state, old)

    def test_reactivation_is_idempotent_and_preserves_live_learning(self):
        self.activate()
        profile = PlayerProfile.objects.get(pk=named_profile_id('example_player'))
        profile.state['moves'] = ['e2e4', 'e7e5']; profile.save()
        self.assertFalse(self.activate()['created'])
        profile.refresh_from_db()
        self.assertEqual(profile.state['moves'], ['e2e4', 'e7e5'])
        self.model['changed_artifact'] = True; self.write_artifacts()
        with self.assertRaisesMessage(ValueError, 'already has a different'):
            self.activate()
        profile.refresh_from_db()
        self.assertEqual(profile.state['moves'], ['e2e4', 'e7e5'])

    def test_incomplete_mismatched_or_wrong_count_artifacts_cannot_activate(self):
        for change in ({'complete_dataset': False}, {'dataset_fingerprint': 'other'},
                       {'username': 'another_player'}):
            original = dict(self.model)
            self.model.update(change); self.write_artifacts()
            with self.assertRaises(ValueError):
                self.activate()
            self.model = original
        self.report['summary']['total_positions'] = 20; self.write_artifacts()
        with self.assertRaises(ValueError):
            self.activate()
        self.assertFalse(self.config.exists())
        self.assertEqual(PlayerProfile.objects.count(), 0)

    def test_in_progress_fit_and_evaluation_checkpoint_cannot_be_activated(self):
        self.report['training_complete'] = False; self.write_artifacts()
        with self.assertRaisesMessage(ValueError, 'completed fit'):
            self.activate()
        self.report['training_complete'] = True
        self.model['purpose'] = 'frozen_training_80_percent'; self.write_artifacts()
        with self.assertRaisesMessage(ValueError, 'completed fit'):
            self.activate()
        self.assertEqual(PlayerProfile.objects.count(), 0)

    def test_missing_or_broken_local_configuration_does_not_break_the_app(self):
        self.assertIsNone(active_username(self.config))
        for value in ('{', '{}', '[]', '{"username":false}', '{"username":"bad name"}'):
            self.config.write_text(value)
            with self.assertLogs('chess_tutor.personal_profile', level='WARNING'):
                self.assertIsNone(active_username(self.config))

    def test_changed_prior_cannot_activate_or_update_an_existing_model(self):
        with patch('chess_tutor.deployment.current_maia_fingerprint', return_value='different-prior'):
            with self.assertRaisesMessage(ValueError, 'different Maia runtime'):
                self.activate()
        self.activate()
        with patch('chess_tutor.views.active_username', return_value='example_player'), \
                patch('chess_tutor.views.current_maia_fingerprint', return_value='different-prior'):
            state = self.client.get('/api/state/').json()
            self.assertFalse(state['ready'])
            response = self.client.post('/api/action/', data=json.dumps(
                {'action': 'move', 'move': 'e4', 'revision': state['revision']}),
                content_type='application/json')
            self.assertEqual(response.status_code, 503)
            self.assertEqual(self.client.get('/api/state/').json(), state)
