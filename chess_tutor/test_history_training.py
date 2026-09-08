"""Offline history behavior with artificial games and explicit synthetic priors."""
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chess

from scripts.train_history import (
    PriorCache, atomic_json, cached_games, main, precompute_priors, run_training, selected_games,
)
from .chesscom_import import iter_training_examples
from .player_policy import PersonalPolicy


def uniform(board):
    moves = sorted(move.uci() for move in board.legal_moves)
    return dict.fromkeys(moves, 1 / len(moves))


def game(index, color="white", moves="1. e4 e5 2. Nf3 Nc6 *", time_class="rapid"):
    return {"url": f"https://example.invalid/game/{index}", "pgn": moves,
            "timestamp": index, "user_color": color, "time_class": time_class}


class HistoryTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fingerprint = {"test_prior": "uniform-v2"}
        self.quiet = patch("builtins.print")
        self.quiet.start()
        self.addCleanup(self.quiet.stop)

    def run_fit(self, games, directory="fit", provider=uniform, **kwargs):
        return run_training(games, "Test_Player", self.root / directory, provider,
                            policy_fingerprint=self.fingerprint, **kwargs)

    def test_split_is_chronological_by_game_and_report_scores_frozen_model(self):
        games = [game(5, moves="1. d4 d5 2. c4 e6 *", time_class="blitz"),
                 game(1), game(4, color="black"), game(2), game(3)]
        report = self.run_fit(games)
        split = json.loads((self.root / "fit/split_model.json").read_text())
        deployed = json.loads((self.root / "fit/deployment_model.json").read_text())
        frozen = PersonalPolicy(split["policy"])
        expected_prior = expected_personal = 0
        examples = list(iter_training_examples(games[0]))
        for example in examples:
            prior = uniform(example["board"])
            personal = frozen.distribution(example["board"], prior)
            expected_prior -= math.log(prior[example["move_uci"]])
            expected_personal -= math.log(personal[example["move_uci"]])
        summary = report["summary"]
        self.assertEqual(summary["train_games"], 4)
        self.assertEqual(summary["test_games"], 1)
        self.assertEqual(summary["train_positions"], 8)
        self.assertEqual(summary["test_positions"], 2)
        self.assertEqual(summary["total_positions"], 10)
        self.assertEqual(split["policy"]["samples"], 8)
        self.assertEqual(deployed["policy"]["samples"], 10)
        self.assertEqual(report["fit_epochs"], 3)
        self.assertTrue(report["training_complete"])
        self.assertEqual(summary, deployed["summary"])
        self.assertIsNotNone(summary["trained_at"])
        self.assertAlmostEqual(summary["prior_log_loss"], expected_prior / 2)
        self.assertAlmostEqual(summary["personal_log_loss"], expected_personal / 2)
        self.assertEqual(report["by_time_class"]["blitz"]["observations"], 2)
        self.assertNotIn("rapid", report["by_time_class"])
        # Final live replay is chronological, independent of the epoch shuffles.
        self.assertEqual(deployed["policy"]["replay"][-1]["chosen"], "c2c4")

    def test_resume_replays_only_partial_game_and_matches_uninterrupted_fit(self):
        games = [game(index) for index in range(5)]
        calls = 0
        def interrupted(board):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("temporary inference interruption")
            return uniform(board)
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.run_fit(games, provider=interrupted)
        checkpoint = json.loads((self.root / "fit/checkpoint.json").read_text())
        self.assertEqual(checkpoint["policy"]["samples"], 2)
        self.assertEqual(checkpoint["cursor"], 1)
        self.run_fit(games)
        self.run_fit(games, "clean")
        resumed = json.loads((self.root / "fit/deployment_model.json").read_text())
        clean = json.loads((self.root / "clean/deployment_model.json").read_text())
        self.assertEqual(resumed["policy"], clean["policy"])
        # Running a completed checkpoint does no inference or training again.
        self.run_fit(games, provider=lambda board: self.fail("completed run queried prior"))

    def test_checkpoint_identity_prevents_username_source_and_runtime_mixups(self):
        games = [game(1), game(2)]
        self.run_fit(games)
        original = (self.root / "fit/checkpoint.json").read_bytes()
        changed = copy.deepcopy(games)
        changed[0]["pgn"] = "1. d4 d5 *"
        with self.assertRaisesRegex(ValueError, "different username, source games"):
            self.run_fit(changed)
        with self.assertRaises(ValueError):
            run_training(games, "Another_User", self.root / "fit", uniform,
                         policy_fingerprint=self.fingerprint)
        with self.assertRaises(ValueError):
            run_training(games, "Test_Player", self.root / "fit", uniform,
                         policy_fingerprint={"test_prior": "changed"})
        self.assertEqual(original, (self.root / "fit/checkpoint.json").read_bytes())

    def test_parser_failure_skips_whole_game_and_smoke_subset_is_not_full(self):
        games = [game(index) for index in range(5)]
        def broken_iterator(record):
            if record["url"] == games[1]["url"]:
                yield next(iter_training_examples(record))
                raise ValueError("malformed trailing PGN")
            yield from iter_training_examples(record)
        report = self.run_fit(games, iterator=broken_iterator, max_games=4)
        self.assertFalse(report["complete_dataset"])
        self.assertEqual(report["summary"]["total_positions"], 6)
        self.assertEqual(report["summary"]["games_total"], 3)
        self.assertEqual(report["summary"]["train_games"], 2)
        self.assertTrue(any(row["url"] == games[1]["url"] for row in report["skipped"]))
        deployed = json.loads((self.root / "fit/deployment_model.json").read_text())
        self.assertEqual(deployed["policy"]["samples"], 6)

    def test_one_game_has_no_heldout_metric_and_illegal_source_owner_is_rejected(self):
        report = self.run_fit([game(1)])
        self.assertEqual(report["prediction_result"], "no_heldout_observations")
        self.assertEqual(report["summary"]["test_games"], 0)
        self.assertIsNone(report["summary"]["personal_log_loss"])
        bad = game(2)
        bad["white"] = {"username": "Other_Player"}
        with self.assertRaisesRegex(ValueError, "another username"):
            selected_games([bad], "Test_Player")


class PriorCacheTests(unittest.TestCase):
    def test_cache_preserves_full_history_and_separates_versions_and_players(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "priors.sqlite3"
            board = chess.Board()
            for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
                board.push_uci(move)
            clone = chess.Board(board.fen())
            with PriorCache(path, "Test_Player", {"version": "v2"}) as cache:
                self.assertNotEqual(cache.key(board), cache.key(clone))
                cache.put(board, uniform(board))
                self.assertEqual(cache.get(board), uniform(board))
                self.assertIsNone(cache.get(clone))
            with PriorCache(path, "test_player", {"version": "other"}) as cache:
                self.assertIsNone(cache.get(board))
            with self.assertRaisesRegex(ValueError, "another username"):
                PriorCache(path, "Other_Player", {"version": "v2"})

    def test_parallel_precompute_reuses_cache_and_closes_all_providers(self):
        calls, closed = [], []
        class Provider:
            def probabilities(self, board):
                calls.append(board.fen())
                return uniform(board)
            def close(self):
                closed.append(self)
        games = [game(1), game(2), game(3, "black")]
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            with PriorCache(Path(directory) / "priors.sqlite3", "Test_Player", {"v": 2}) as cache:
                result = precompute_priors(games, cache, directory, "Test_Player", workers=2,
                                           provider_factory=Provider)
                initial_calls = len(calls)
                self.assertEqual(result["observations_seen"], 6)
                self.assertEqual(initial_calls, 4)
                self.assertTrue(closed)
                precompute_priors(games, cache, directory, "Test_Player", workers=2,
                                  provider_factory=Provider)
                self.assertEqual(len(calls), initial_calls)


class LocalResumeTests(unittest.TestCase):
    def test_checkpoint_fast_encoder_is_byte_identical_to_streaming_encoder(self):
        board = chess.Board()
        policy = PersonalPolicy()
        policy.learn(board, uniform(board), "e2e4")
        state = {"policy": policy.dump(), "unicode": "café", "float": -0.0,
                 "nested": [True, None, {"number": 1e-12}]}
        expected = io.StringIO()
        json.dump(state, expected, sort_keys=True, allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            atomic_json(path, state)
            self.assertEqual(path.read_text(), expected.getvalue())

    def test_fit_only_reads_verified_local_archives_without_fetch_or_prior_precompute(self):
        raw = {"url": "https://example.invalid/1", "rules": "chess", "end_time": 1,
               "white": {"username": "Test_Player", "result": "win"},
               "black": {"username": "Opponent", "result": "resigned"},
               "pgn": "1. e4 e5 *", "time_class": "rapid"}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            player = base / "test_player"
            player.mkdir()
            payload = json.dumps({"games": [raw]}).encode()
            (player / "2020-01.json").write_bytes(payload)
            manifest = {"username": "test_player", "status": "complete", "archive_count": 1,
                        "archives_completed": 1, "accepted_games": 1, "human_moves": 1,
                        "archives": [{"file": "2020-01.json", "sha256": hashlib.sha256(payload).hexdigest()}]}
            atomic_json(player / "manifest.json", manifest)
            with patch("scripts.train_history.fetch_games") as fetch, \
                    patch("scripts.train_history.precompute_priors") as precompute, \
                    patch("scripts.train_history.maia_fingerprint", return_value={"v": 2}), \
                    patch("scripts.train_history.run_training") as train:
                main(["--username", "Test_Player", "--cache-dir", directory, "--fit-only"])
                fetch.assert_not_called()
                precompute.assert_not_called()
                self.assertEqual(train.call_args.args[0][0]["url"], raw["url"])
            (player / "2020-01.json").write_bytes(payload + b" ")
            with self.assertRaisesRegex(ValueError, "changed after import"):
                cached_games("Test_Player", base)

    def test_stage_flags_are_mutually_exclusive(self):
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            main(["--username", "Test_Player", "--fit-only", "--priors-only"])


if __name__ == "__main__":
    unittest.main()
