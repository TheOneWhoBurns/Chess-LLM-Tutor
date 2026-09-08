"""Coaching request persistence, snapshot isolation, and safe retries."""
from copy import deepcopy
from datetime import timedelta
import json
import uuid
from unittest.mock import patch

from django.db import OperationalError
from django.test import Client, TestCase
from django.utils import timezone

from .astra import AstraUnavailable
from .chat_views import claim_exchange, fresh_analysis
from .models import ChatExchange, PlayerProfile


class ChatTests(TestCase):
    def setUp(self):
        patches = [
            patch('chess_tutor.views.active_username', return_value=None),
            patch('chess_tutor.chat_views.astra.configuration_status', return_value={
                'ready': True, 'model': 'gpt-6-astra', 'error': None}),
            patch('chess_tutor.chat_views.fresh_analysis', return_value={
                'engine': 'Stockfish test double', 'perspective': 'white',
                'moves': [{'move': 'e4', 'score_cp': 25, 'mate': None, 'depth': 12, 'nodes': 60000}]}),
            patch('chess_tutor.chat_views.astra.answer', return_value='Develop your pieces toward the center.'),
        ]
        mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        _, self.config, self.analysis, self.answer = mocks
        self.client.get('/')
        self.profile = PlayerProfile.objects.get(pk=self.client.session['nemesis_profile'])
        self.body = {'message': 'How should I improve?', 'revision': self.profile.revision,
                     'request_id': str(uuid.uuid4())}

    def send(self, **changes):
        return self.client.post('/api/chat/', data=json.dumps({**self.body, **changes}),
                                content_type='application/json')

    def test_chat_persists_and_does_not_mutate_game_or_policy(self):
        before = deepcopy(self.profile.state)
        self.assertEqual(self.client.get('/api/chat/').json()['messages'], [])
        response = self.send()
        self.assertEqual(response.status_code, 200, response.content)
        data = response.json()
        self.assertEqual([m['role'] for m in data['messages']], ['user', 'assistant'])
        self.assertEqual(data['messages'][1]['model'], 'gpt-6-astra')
        self.assertIn('Move 1', data['messages'][1]['context_label'])
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.state, before)
        self.assertEqual(self.profile.revision, self.body['revision'])
        restored = Client(); restored.cookies = self.client.cookies
        self.assertEqual(restored.get('/api/chat/').json(), data)
        self.assertEqual(self.client.get('/api/export/?format=json').json()['chat'], data['messages'])

    def test_next_message_receives_previous_conversation(self):
        self.send()
        response = self.send(request_id=str(uuid.uuid4()), message='Why?')
        self.assertEqual(response.status_code, 200)
        message, history, context = self.answer.call_args.args
        self.assertEqual(message, 'Why?')
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['content'], self.body['message'])
        self.assertEqual(len(response.json()['messages']), 4)

    def test_completed_retry_is_cached_even_after_board_or_config_changes(self):
        first = self.send().json()['messages']
        self.profile.revision += 1
        self.profile.save(update_fields=['revision'])
        self.config.return_value = {'ready': False, 'model': 'gpt-6-astra', 'error': 'Configure a key.'}
        retry = self.send()
        self.assertEqual(retry.status_code, 200)
        self.assertEqual(retry.json()['messages'], first)
        self.answer.assert_called_once()
        self.assertEqual(self.send(message='A different message').status_code, 409)

    def test_failed_call_can_retry_same_id_without_duplicate_messages(self):
        self.answer.side_effect = AstraUnavailable('Astra timed out.', status_code=504, code='timeout')
        failed = self.send()
        self.assertEqual(failed.status_code, 504)
        self.assertEqual(failed.json()['messages'], [])
        self.assertEqual(ChatExchange.objects.get().status, 'failed')
        self.answer.side_effect = None
        self.assertEqual(self.send().status_code, 200)
        self.assertEqual(ChatExchange.objects.count(), 1)
        self.assertEqual(ChatExchange.objects.get().status, 'complete')

    def test_stale_revision_is_rejected_before_external_call(self):
        self.profile.revision += 1
        self.profile.save(update_fields=['revision'])
        response = self.send()
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()['error_code'], 'stale_revision')
        self.assertEqual(ChatExchange.objects.count(), 0)
        self.answer.assert_not_called()
        self.analysis.assert_not_called()

    def test_board_advance_during_answer_preserves_new_game_state(self):
        advanced = deepcopy(self.profile.state)
        advanced.update(moves=['e2e4', 'e7e5'])
        def finish(*args):
            PlayerProfile.objects.filter(pk=self.profile.pk).update(state=advanced, revision=1)
            return 'At the start, develop your pieces.'
        self.answer.side_effect = finish
        response = self.send()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['revision'], 1)
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.state, advanced)
        self.assertEqual(ChatExchange.objects.get().board_revision, 0)
        self.assertIn('Move 1', response.json()['messages'][1]['context_label'])

    def test_pending_exchange_rejects_duplicate_and_second_request(self):
        claim_exchange(self.profile, uuid.UUID(self.body['request_id']), self.body['message'], self.body['revision'])
        for request_id in (self.body['request_id'], str(uuid.uuid4())):
            response = self.send(request_id=request_id)
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()['error_code'], 'chat_busy')
            self.assertEqual(ChatExchange.objects.get().status, 'pending')
        self.answer.assert_not_called()

    def test_expired_worker_cannot_overwrite_or_fail_new_claim(self):
        def reclaim(*args):
            ChatExchange.objects.update(updated_at=timezone.now() - timedelta(seconds=121))
            claim_exchange(self.profile, uuid.UUID(self.body['request_id']), self.body['message'], self.body['revision'])
            return 'Old worker answer'
        self.answer.side_effect = reclaim
        response = self.send()
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()['error_code'], 'request_expired')
        row = ChatExchange.objects.get()
        self.assertEqual(row.status, 'pending')
        self.assertEqual(row.assistant_message, '')

    def test_interrupted_request_can_be_reclaimed(self):
        claim_exchange(self.profile, uuid.UUID(self.body['request_id']), self.body['message'], self.body['revision'])
        ChatExchange.objects.update(updated_at=timezone.now() - timedelta(seconds=121))
        self.assertEqual(self.send().status_code, 200)
        self.assertEqual(ChatExchange.objects.count(), 1)

    def test_missing_configuration_does_not_call_or_reserve(self):
        self.config.return_value = {'ready': False, 'model': 'gpt-6-astra', 'error': 'Configure a key.'}
        response = self.send()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()['error_code'], 'not_configured')
        self.assertEqual(ChatExchange.objects.count(), 0)
        self.answer.assert_not_called()

    def test_busy_database_still_returns_safe_retry_response(self):
        with patch('chess_tutor.chat_views.claim_exchange', side_effect=OperationalError('database is locked')):
            with patch.object(PlayerProfile, 'refresh_from_db', side_effect=OperationalError('database is locked')):
                response = self.send()
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()['error_code'], 'chat_busy')
        self.assertNotIn('messages', response.json())
        self.answer.assert_not_called()

    def test_resigned_game_does_not_run_engine_analysis(self):
        state = deepcopy(self.profile.state)
        state['result'] = '0-1'
        with patch('chess_tutor.chat_views.engine_room') as engine:
            self.assertEqual(fresh_analysis(state), {'game_over': True, 'result': '0-1'})
            engine.assert_not_called()

    def test_input_csrf_and_methods(self):
        for changes in ({'message': ''}, {'message': 'a' * 4001}, {'message': []},
                        {'revision': True}, {'revision': -1}, {'request_id': 'bad'}, {'request_id': None}):
            self.assertEqual(self.send(**changes).status_code, 400)
        for body in ('{', '[]', 'null'):
            self.assertEqual(self.client.post('/api/chat/', data=body, content_type='application/json').status_code, 400)
        self.assertEqual(self.client.put('/api/chat/').status_code, 405)
        csrf_client = Client(enforce_csrf_checks=True); csrf_client.get('/')
        self.assertEqual(csrf_client.post('/api/chat/', data=json.dumps(self.body),
                        content_type='application/json').status_code, 403)
        self.answer.assert_not_called()

    def test_profiles_have_separate_chat_history(self):
        self.send()
        self.assertEqual(Client().get('/api/chat/').json()['messages'], [])
