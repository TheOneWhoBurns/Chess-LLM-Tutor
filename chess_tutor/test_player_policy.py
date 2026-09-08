"""Behavioral tests with real legal moves and explicit synthetic policy priors."""
import copy
import json
import math
import unittest

import chess
import numpy as np

from .player_policy import (
    MAX_FEATURE_CACHE, MAX_FIT_BATCH, MAX_REPLAY, PersonalPolicy, _FEATURE_CACHE,
    _compute_feature_matrix, _feature_matrix,
)


def uniform_prior(board):
    moves = [move.uci() for move in board.legal_moves]
    return {move: 1 / len(moves) for move in moves}


class FeatureMatrixCacheTests(unittest.TestCase):
    def setUp(self):
        _FEATURE_CACHE.clear()

    def test_cached_features_are_exact_and_cannot_be_mutated(self):
        positions = [chess.Board()] + [chess.Board(fen) for fen in (
            "r3k2r/ppp2ppp/2n5/3pp3/8/2N5/PPP2PPP/R3K2R w KQkq - 0 10",
            "7k/P7/8/8/8/8/8/7K w - - 0 1",
            "7K/8/8/8/8/8/p7/7k b - - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
        )]
        for board in positions:
            with self.subTest(fen=board.fen()):
                moves = sorted(move.uci() for move in board.legal_moves)
                expected = _compute_feature_matrix(board, moves)
                actual = _feature_matrix(board, moves)
                np.testing.assert_array_equal(actual, expected)
                self.assertIs(_feature_matrix(board.copy(), moves), actual)
                with self.assertRaises(ValueError):
                    actual[0, 0] = 99
                with self.assertRaises(ValueError):
                    actual.setflags(write=True)

    def test_counters_and_row_order_are_retained_but_history_is_not_required(self):
        board = chess.Board()
        for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push_uci(move)
        before = (board.fen(), list(board.move_stack), board.is_repetition(2))
        moves = sorted(move.uci() for move in board.legal_moves)
        actual = _feature_matrix(board, moves)
        self.assertIs(_feature_matrix(chess.Board(board.fen()), moves), actual)
        self.assertEqual(before, (board.fen(), list(board.move_stack), board.is_repetition(2)))
        fresh_counters = board.copy()
        fresh_counters.halfmove_clock = 0
        fresh_counters.fullmove_number = 1
        reset = _feature_matrix(fresh_counters, moves)
        np.testing.assert_array_equal(reset, _compute_feature_matrix(fresh_counters, moves))
        self.assertFalse(np.array_equal(reset, actual))
        np.testing.assert_array_equal(_feature_matrix(board, list(reversed(moves))), actual[::-1])

    def test_cache_evicts_least_recently_used_positions_at_its_bound(self):
        board = chess.Board()
        moves = sorted(move.uci() for move in board.legal_moves)
        oldest = _feature_matrix(board, moves)
        board.fullmove_number = 2
        second = _feature_matrix(board, moves)
        for move_number in range(3, MAX_FEATURE_CACHE + 1):
            board.fullmove_number = move_number
            _feature_matrix(board, moves)
        self.assertEqual(len(_FEATURE_CACHE), MAX_FEATURE_CACHE)
        board.fullmove_number = 1
        self.assertIs(_feature_matrix(board, moves), oldest)
        board.fullmove_number = MAX_FEATURE_CACHE + 1
        _feature_matrix(board, moves)
        self.assertEqual(len(_FEATURE_CACHE), MAX_FEATURE_CACHE)
        board.fullmove_number = 1
        self.assertIs(_feature_matrix(board, moves), oldest)
        board.fullmove_number = 2
        self.assertIsNot(_feature_matrix(board, moves), second)
        self.assertEqual(len(_FEATURE_CACHE), MAX_FEATURE_CACHE)


class PersonalPolicyTests(unittest.TestCase):
    def test_offline_batches_learn_choices_without_counting_later_epochs_again(self):
        board = chess.Board()
        prior = uniform_prior(board)
        examples = [{"board": board, "prior": prior, "chosen_uci": "e2e4"}] * MAX_FIT_BATCH
        model = PersonalPolicy()
        result = model.fit_batch(examples)
        self.assertEqual(result, {"samples_before": 0, "samples": MAX_FIT_BATCH,
                                  "examples": MAX_FIT_BATCH, "steps": 1})
        first_probability = model.distribution(board, prior)["e2e4"]
        retained = copy.deepcopy(model.replay)
        for _ in range(2):
            result = model.fit_batch(examples, count_observations=False)
            self.assertEqual(result["samples_before"], MAX_FIT_BATCH)
            self.assertEqual(result["samples"], MAX_FIT_BATCH)
        self.assertEqual(model.replay, retained)
        self.assertGreater(first_probability, prior["e2e4"])
        self.assertGreater(model.distribution(board, prior)["e2e4"], first_probability)
        restored = PersonalPolicy(json.loads(json.dumps(model.dump())))
        self.assertEqual(restored.dump(), model.dump())
        self.assertEqual(restored.distribution(board, prior), model.distribution(board, prior))

    def test_single_example_batch_uses_exact_online_optimizer(self):
        board = chess.Board()
        prior = uniform_prior(board)
        online, offline = PersonalPolicy(), PersonalPolicy()
        online.learn(board, prior, "e2e4")
        offline.fit_batch([{"board": board, "prior": prior, "chosen_uci": "e2e4"}], steps=10)
        self.assertEqual(online.dump(), offline.dump())

    def test_shuffled_batches_can_restore_chronological_bounded_replay(self):
        examples = []
        for index in range(MAX_REPLAY + 4):
            board = chess.Board()
            board.fullmove_number = index + 1
            examples.append({"board": board, "prior": uniform_prior(board), "chosen_uci": "e2e4"})
        model = PersonalPolicy()
        shuffled = list(reversed(examples))
        for start in range(0, len(shuffled), MAX_FIT_BATCH):
            model.fit_batch(shuffled[start:start + MAX_FIT_BATCH])
        before = model.dump()
        model.replace_replay(examples[-MAX_REPLAY:])
        after = model.dump()
        self.assertEqual([row["fen"] for row in after["replay"]],
                         [example["board"].fen() for example in examples[-MAX_REPLAY:]])
        for key in before.keys() - {"replay"}:
            self.assertEqual(before[key], after[key])
        restored = PersonalPolicy(json.loads(json.dumps(after)))
        next_example = examples[-1]
        for policy in (model, restored):
            policy.learn(next_example["board"], next_example["prior"], "d2d4")
        self.assertEqual(model.dump(), restored.dump())

    def test_offline_inputs_are_all_validated_before_changing_model_state(self):
        board = chess.Board()
        prior = uniform_prior(board)
        example = {"board": board, "prior": prior, "chosen_uci": "e2e4"}
        model = PersonalPolicy()
        model.learn(board, prior, "d2d4")
        before = model.dump()
        invalid_examples = [
            [], (), "examples", [example] * (MAX_FIT_BATCH + 1),
            [example, None], [example, {}],
            [example, {**example, "board": None}],
            [example, {**example, "prior": {}}],
            [example, {**example, "prior": {**prior, "e2e4": float("nan")}}],
            [example, {**example, "chosen_uci": "e2e5"}],
        ]
        for invalid in invalid_examples:
            with self.subTest(examples=invalid):
                with self.assertRaises(ValueError):
                    model.fit_batch(invalid)
                self.assertEqual(model.dump(), before)
        for options in ({"steps": 0}, {"steps": -1}, {"steps": True}, {"steps": 1.5},
                        {"count_observations": 1}, {"count_observations": None}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    model.fit_batch([example], **options)
                self.assertEqual(model.dump(), before)
        for invalid in ([], [example, example], [{**example, "chosen_uci": "e2e5"}]):
            with self.assertRaises(ValueError):
                model.replace_replay(invalid)
            self.assertEqual(model.dump(), before)

    def test_new_profile_uses_supplied_prior_without_invented_preferences(self):
        board = chess.Board()
        prior = uniform_prior(board)
        prior["e2e4"] = 4
        expected = {move: probability / sum(prior.values()) for move, probability in prior.items()}
        model = PersonalPolicy()
        actual = model.distribution(board, prior)
        self.assertEqual(set(actual), set(prior))
        for move in prior:
            self.assertAlmostEqual(actual[move], expected[move])
        self.assertEqual(model.samples, 0)
        self.assertEqual(model.adaptation_weight, 0)

    def test_demonstrated_choice_improves_future_predictions_and_round_trips(self):
        board = chess.Board()
        prior = uniform_prior(board)
        model = PersonalPolicy()
        diagnostics = [model.learn(board, prior, "e2e4") for _ in range(48)]
        learned = model.distribution(board, prior)
        self.assertGreater(learned["e2e4"], 2 * prior["e2e4"])
        self.assertEqual(max(learned, key=learned.get), "e2e4")
        self.assertAlmostEqual(diagnostics[0]["prior_log_loss"], diagnostics[0]["personal_log_loss"])
        self.assertLess(sum(item["personal_log_loss"] for item in diagnostics[-12:]),
                        sum(item["prior_log_loss"] for item in diagnostics[-12:]))
        restored = PersonalPolicy(json.loads(json.dumps(model.dump())))
        self.assertEqual(restored.samples, 48)
        for move, probability in learned.items():
            self.assertAlmostEqual(restored.distribution(board, prior)[move], probability)
        # Restored replay must support the same next update, not just inference.
        restored.learn(board, prior, "d2d4")
        model.learn(board, prior, "d2d4")
        for move, probability in model.distribution(board, prior).items():
            self.assertAlmostEqual(restored.distribution(board, prior)[move], probability)

    def test_diagnostics_score_the_prediction_before_observing_the_move(self):
        board = chess.Board()
        prior = uniform_prior(board)
        model = PersonalPolicy()
        for _ in range(12):
            model.learn(board, prior, "g1f3")
        before = model.distribution(board, prior)
        samples_before = model.samples
        result = model.learn(board, prior, "d2d4")
        self.assertAlmostEqual(result["personal_probability"], before["d2d4"])
        self.assertAlmostEqual(result["personal_log_loss"], -math.log(before["d2d4"]))
        self.assertAlmostEqual(result["prior_probability"], prior["d2d4"])
        self.assertEqual(result["top_prediction"], max(before, key=before.get))
        self.assertEqual(result["samples_before"], samples_before)
        self.assertEqual(result["samples"], model.samples)

    def test_different_demonstrations_produce_different_personal_preferences(self):
        board = chess.Board()
        prior = uniform_prior(board)
        first, second = PersonalPolicy(), PersonalPolicy()
        for _ in range(24):
            first.learn(board, prior, "e2e4")
            second.learn(board, prior, "d2d4")
        first_distribution = first.distribution(board, prior)
        second_distribution = second.distribution(board, prior)
        self.assertGreater(first_distribution["e2e4"], second_distribution["e2e4"])
        self.assertGreater(second_distribution["d2d4"], first_distribution["d2d4"])

    def test_special_moves_and_black_positions_preserve_the_callers_board(self):
        cases = [
            ("r3k2r/ppp2ppp/2n5/3pp3/8/2N5/PPP2PPP/R3K2R w KQkq - 0 10", "e1g1"),
            ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8n"),
            ("7K/8/8/8/8/8/p7/7k b - - 0 1", "a2a1r"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6"),
        ]
        model = PersonalPolicy()
        for fen, choice in cases:
            with self.subTest(move=choice):
                board = chess.Board(fen)
                prior = uniform_prior(board)
                before = (board.fen(en_passant="fen"), list(board.move_stack))
                model.distribution(board, prior)
                model.learn(board, prior, choice)
                distribution = model.distribution(board, prior)
                self.assertEqual(set(distribution), set(prior))
                self.assertAlmostEqual(sum(distribution.values()), 1)
                self.assertTrue(all(math.isfinite(value) and value > 0 for value in distribution.values()))
                self.assertEqual(before, (board.fen(en_passant="fen"), list(board.move_stack)))
        # Preserve a nonempty repetition history as well as isolated FEN positions.
        board = chess.Board()
        for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push_uci(move)
        before = (board.fen(), list(board.move_stack), board.is_repetition(2))
        model.learn(board, uniform_prior(board), "e2e4")
        self.assertEqual(before, (board.fen(), list(board.move_stack), board.is_repetition(2)))

    def test_invalid_priors_or_moves_do_not_change_saved_model(self):
        board = chess.Board()
        prior = uniform_prior(board)
        invalid = [
            {}, {**prior, "e2e5": 0.1}, {**prior, "e2e4": -1},
            {**prior, "e2e4": float("nan")}, {**prior, "e2e4": True},
            dict.fromkeys(prior, 0),
        ]
        model = PersonalPolicy()
        before = model.dump()
        for broken in invalid:
            with self.assertRaises(ValueError):
                model.learn(board, broken, "e2e4")
            with self.assertRaises(ValueError):
                model.distribution(board, broken)
        with self.assertRaises(ValueError):
            model.learn(board, prior, "e2e5")
        self.assertEqual(before, model.dump())

    def test_saved_schema_rejects_old_models_invalid_weights_and_illegal_replay(self):
        board = chess.Board()
        model = PersonalPolicy()
        model.learn(board, uniform_prior(board), "e2e4")
        good = model.dump()
        invalid = [None] * 6
        invalid[0] = {**good, "version": 1}
        invalid[1] = {**good, "samples": True}
        invalid[2] = {**good, "w2": [0]}
        invalid[3] = {**good, "b2": float("inf")}
        invalid[4] = copy.deepcopy(good)
        invalid[4]["replay"][0]["chosen"] = "e2e5"
        invalid[5] = {**good, "samples": 0}
        for broken in invalid:
            with self.assertRaises(ValueError):
                PersonalPolicy(broken)
        detached = model.dump()
        detached["replay"][0]["prior"]["e2e4"] = 99
        self.assertEqual(good, model.dump())

    def test_replay_is_bounded_and_restarted_models_remain_trainable(self):
        # A forced reply keeps this a fast long-history persistence test.
        board = chess.Board("6Rk/8/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(board.legal_moves.count(), 1)
        prior = uniform_prior(board)
        choice = next(iter(prior))
        model = PersonalPolicy()
        for _ in range(MAX_REPLAY + 2):
            model.learn(board, prior, choice)
        dumped = model.dump()
        self.assertEqual(len(dumped["replay"]), MAX_REPLAY)
        restored = PersonalPolicy(json.loads(json.dumps(dumped)))
        restored.learn(board, prior, choice)
        self.assertEqual(restored.samples, MAX_REPLAY + 3)
        self.assertEqual(restored.distribution(board, prior), prior)

    def test_terminal_position_has_no_policy_or_trainable_choice(self):
        board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(board.is_checkmate())
        model = PersonalPolicy()
        self.assertEqual(model.distribution(board, {}), {})
        with self.assertRaises(ValueError):
            model.learn(board, {}, "h8g8")


if __name__ == "__main__":
    unittest.main()
