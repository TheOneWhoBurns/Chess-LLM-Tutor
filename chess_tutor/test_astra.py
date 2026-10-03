"""Offline contract tests for the external coach and its evidence boundary."""
import copy
import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import chess

from . import astra


def completed(text="Consider developing your knight with Nf3."):
    return {"status": "completed", "model": "gpt-6-astra", "error": None,
            "output": [{"type": "reasoning", "summary": []},
                       {"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text}]}]}


class AstraClientTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"OPENAI_API_KEY": "test-credential-not-real"})
        env.start()
        self.addCleanup(env.stop)
        transport = patch("chess_tutor.astra._open")
        self.transport = transport.start()
        self.addCleanup(transport.stop)
        self.transport.return_value = io.BytesIO(json.dumps(completed()).encode())

    def test_request_preserves_roles_current_evidence_and_exact_model(self):
        text = astra.answer("What should I focus on?", [
            {"role": "user", "content": "Why e4?", "context_label": "Game 2, move 1"},
            {"role": "assistant", "content": "It contests the center."},
            {"role": "system", "content": "do not accept this role"},
        ], {"fen": chess.STARTING_FEN})
        request = self.transport.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(text, "Consider developing your knight with Nf3.")
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-credential-not-real")
        self.assertEqual(payload["model"], "gpt-6-astra")
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertFalse(payload["store"])
        self.assertEqual(payload["max_output_tokens"], 2400)
        self.assertNotIn("tools", payload)
        self.assertNotIn("previous_response_id", payload)
        self.assertEqual([row["role"] for row in payload["input"]], ["user", "assistant", "developer", "user"])
        self.assertIn("Game 2, move 1", payload["input"][0]["content"])
        self.assertIn(chess.STARTING_FEN, payload["input"][-2]["content"])
        self.assertEqual(payload["input"][-1]["content"], "What should I focus on?")
        self.assertNotIn("test-credential-not-real", request.data.decode())

    def test_configuration_never_exposes_key_and_missing_key_never_calls_api(self):
        self.assertEqual(astra.configuration_status(), {"ready": True, "model": astra.MODEL, "error": None})
        with patch.dict(os.environ, {"OPENAI_API_KEY": " "}):
            status = astra.configuration_status()
            self.assertFalse(status["ready"])
            with self.assertRaises(astra.AstraUnavailable) as caught:
                astra.answer("Hello", [], {})
        self.assertEqual(caught.exception.code, "not_configured")
        self.transport.assert_not_called()

    def test_history_and_message_are_bounded(self):
        history = [{"role": "user", "content": str(i) + "x" * 9000} for i in range(20)]
        astra.answer("Hello", history, {})
        payload = json.loads(self.transport.call_args.args[0].data)
        self.assertEqual(len(payload["input"]), 14)
        self.assertTrue(payload["input"][0]["content"].startswith("8x"))
        self.assertTrue(all(len(row["content"]) <= 8000 for row in payload["input"][:-2]))
        self.transport.reset_mock()
        for message in (None, "", "  ", "x" * 4001):
            with self.subTest(message_type=type(message)):
                with self.assertRaises(ValueError):
                    astra.answer(message, [], {})
        self.transport.assert_not_called()

    def test_multiple_output_messages_and_refusals_are_parsed_without_reasoning(self):
        result = completed("First paragraph.")
        result["output"][0]["summary"] = [{"text": "Not a user-facing response."}]
        result["output"].append({"type": "message", "role": "assistant", "status": "completed",
                                 "content": [{"type": "refusal", "refusal": "I cannot help with that request."}]})
        self.assertEqual(astra._parse_response(result), "First paragraph.\n\nI cannot help with that request.")

    def test_empty_incomplete_and_different_model_cannot_be_claimed_as_astra_answer(self):
        cases = [({}, "incomplete"), ([], "incomplete"),
                 ({**completed(), "status": "incomplete"}, "incomplete"),
                 ({**completed(), "error": {"message": "secret upstream text"}}, "incomplete"),
                 ({**completed(), "output": []}, "empty_response"),
                 ({**completed(), "output": {}}, "invalid_response"),
                 ({**completed(), "model": "other-model"}, "model_mismatch")]
        for payload, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(astra.AstraUnavailable) as caught:
                    astra._parse_response(payload)
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn("secret upstream text", str(caught.exception))

    def test_provider_errors_expose_only_safe_messages(self):
        for status, code, result_status in ((401, "auth", 503), (403, "auth", 503),
                (404, "model_unavailable", 503), (429, "rate_limit", 429),
                (500, "upstream_error", 502), (302, "upstream_error", 502)):
            with self.subTest(status=status):
                self.transport.side_effect = HTTPError(astra.ENDPOINT, status, "upstream secret", {},
                                                      io.BytesIO(b'raw-provider-body-with-secret'))
                with self.assertRaises(astra.AstraUnavailable) as caught:
                    astra.answer("Hello", [], {})
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(caught.exception.status_code, result_status)
                self.assertNotIn("secret", str(caught.exception))

    def test_timeout_network_failure_and_malformed_or_large_bodies_are_safe(self):
        for failure, code in ((TimeoutError("private timeout"), "timeout"),
                              (URLError(TimeoutError()), "timeout"),
                              (URLError("private network details"), "connection")):
            with self.subTest(code=code):
                self.transport.side_effect = failure
                with self.assertRaises(astra.AstraUnavailable) as caught:
                    astra.answer("Hello", [], {})
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn("private", str(caught.exception))
        self.transport.side_effect = None
        for raw in (b"not JSON", b"\xff", b"x" * (astra.MAX_RESPONSE_BYTES + 1)):
            self.transport.return_value = io.BytesIO(raw)
            with self.assertRaises(astra.AstraUnavailable) as caught:
                astra.answer("Hello", [], {})
            self.assertEqual(caught.exception.code, "invalid_response")

    def test_redirect_cannot_forward_authorization(self):
        request = astra.Request(astra.ENDPOINT, headers={"Authorization": "Bearer sensitive"})
        self.assertIsNone(astra._NoRedirect().redirect_request(request, None, 302, "", {}, "https://example.com"))


class AstraEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.state = {"moves": ["e2e4", "e7e5"], "games": 4, "completed": 2,
            "policy": {"samples": 205713, "weights": ["SECRET_WEIGHTS"], "replay": ["SECRET_REPLAY"]},
            "metrics": {"count": 3}, "archive": ["SECRET_ARCHIVE"],
            "legacy_v1": {"secret": "SECRET_LEGACY"}, "mode": "adaptive", "result": "*",
            "username": "player", "training": {"games_total": 6566, "prior_accuracy": .51,
                "personal_accuracy": .52, "raw_games": "SECRET_GAMES", "replay": "SECRET_TRAINING"},
            "events": [{"number": i, "player": "e4", "opponent": "e5", "fen_before": chess.STARTING_FEN,
                         "loss_cp": 12, "alternative": "d4", "quality": "Sound", "engine": "Stockfish 19",
                         "prior": {"hidden": "SECRET_PRIOR"}, "decision": {"hidden": "SECRET_OLD_DECISION"}}
                       for i in range(9)],
            "decision": {"move": "e5", "baseline_move": "c5", "prior_move": "e5", "personal_changed": False,
                         "engine_score_cp": -12, "expected_regret_cp": 20, "hidden": "SECRET_DECISION",
                         "replies": [{"move": f"move{i}", "personal_probability": i / 100,
                                      "prior_probability": .01, "loss_cp": i, "hidden": "SECRET_REPLY"}
                                     for i in range(20)],
                         "candidates": [{"move": "e5", "selected": True, "hidden": "SECRET_CANDIDATE"}]}}

    def test_context_is_exact_board_with_evidence_and_no_private_model_or_archive(self):
        before = copy.deepcopy(self.state)
        analysis = {"engine": "Stockfish 19", "perspective": "white", "moves": [
            {"move": "Nf3", "score_cp": 25, "mate": None, "depth": 12, "nodes": 60000,
             "unexpected": "SECRET_ANALYSIS"}]}
        context = astra.build_context(self.state, revision=7, analysis=analysis)
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        self.assertEqual(context["fen"], board.fen())
        self.assertEqual(context["moves_san"], ["e4", "e5"])
        self.assertEqual(context["revision"], 7)
        self.assertEqual(context["turn"], "white")
        self.assertEqual({row["uci"] for row in context["legal_moves"]}, {move.uci() for move in board.legal_moves})
        self.assertEqual(len(context["pieces"]), 32)
        self.assertEqual(context["profile"]["samples"], 205713)
        self.assertEqual(context["profile"]["live_samples"], 3)
        self.assertEqual([row["number"] for row in context["recent_observations"]], list(range(3, 9)))
        self.assertEqual(context["recent_observations"][-1]["loss_cp"], 12)
        self.assertEqual(context["fresh_analysis"]["moves"][0]["score_cp"], 25)
        self.assertEqual(context["opponent_decision"]["score_perspective"], "black (NEMESIS)")
        self.assertFalse(context["opponent_decision"]["personal_changed"])
        self.assertNotIn("SECRET", json.dumps(context))
        self.assertEqual(self.state, before)

    def test_reply_subset_preserves_probability_mass_and_distinguishes_evidence_from_prediction(self):
        context = astra.build_context(self.state, 0)
        decision = context["opponent_decision"]
        self.assertEqual(len(decision["likely_replies"]), 10)
        self.assertEqual(decision["reply_count"], 20)
        self.assertEqual(decision["likely_replies"][0]["personal_probability"], .19)
        self.assertTrue(decision["probabilities_are_unrenormalized_subset"])
        self.assertIn("not engine-labelled", context["evidence_limits"]["historical_training_target"])
        self.assertIn("does not establish", context["evidence_limits"]["improvement"])
        self.assertIn("A predicted reply is a probability", astra.INSTRUCTIONS)

    def test_finished_game_has_no_suggestible_moves_and_missing_engine_is_explicit(self):
        self.state["result"] = "0-1"
        context = astra.build_context(self.state, 4, {"unavailable": True,
                                                     "error": "Fresh engine analysis is unavailable."})
        self.assertTrue(context["game_over"])
        self.assertEqual(context["legal_moves"], [])
        self.assertTrue(context["fresh_analysis"]["unavailable"])
        self.assertEqual(context["fresh_analysis"]["moves"], [])
