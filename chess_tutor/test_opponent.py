"""Opponent selection invariants, independent of downloaded chess engines."""

import os
import threading
from unittest import TestCase, skipUnless
from unittest.mock import MagicMock, patch

import chess
import chess.engine

from .opponent import EngineRoom, EngineUnavailable, MATE, MAX_COST, Stockfish


def row(uci, score, mate=None):
    return {"move": chess.Move.from_uci(uci), "score": score, "mate": mate,
            "depth": 14, "nodes": 10000}


def reply_distribution(board, bad_probability):
    """An exact two-choice distribution within the complete legal move set."""
    result = {move.uci(): 0. for move in board.legal_moves}
    assert "a7a6" in result and "e7e5" in result
    result["a7a6"] = bad_probability
    result["e7e5"] = 1 - bad_probability
    return result


class SearchFixture:
    """Complete legal black replies with one independently scored bad move."""

    name = "Stockfish fixture"

    def __init__(self, roots=None, loss=200):
        self.roots = roots or [row("e2e4", 50), row("d2d4", 20)]
        self.loss = loss
        self.calls = []

    def rank(self, board, count=6, nodes=None):
        self.calls.append(board.fen())
        if not board.move_stack:
            return self.roots
        return sorted(
            [row(move.uci(), -self.loss if move.uci() == "a7a6" else 0)
             for move in board.legal_moves],
            key=lambda item: (-item["score"], item["move"].uci()),
        )


class MaiaFixture:
    info = {"name": "Maia fixture"}

    def __init__(self, bad_probabilities=None):
        self.bad_probabilities = bad_probabilities or {}
        self.calls = []

    def probabilities(self, board):
        preceding = board.peek().uci()
        self.calls.append(preceding)
        return reply_distribution(board, self.bad_probabilities.get(preceding, .1))


class PolicyFixture:
    def __init__(self, bad_probabilities=None, samples=0):
        self.bad_probabilities = bad_probabilities or {}
        self.samples = samples

    def distribution(self, board, prior):
        preceding = board.peek().uci()
        if preceding in self.bad_probabilities:
            return reply_distribution(board, self.bad_probabilities[preceding])
        return dict(prior)


class SelectionTests(TestCase):
    def test_personal_reply_distribution_changes_selection_with_same_search_and_prior(self):
        board = chess.Board()
        room = EngineRoom(SearchFixture(), MaiaFixture())
        neutral, before = room.choose(board, PolicyFixture())
        personal, after = room.choose(board, PolicyFixture({"d2d4": .8}, samples=40))
        self.assertEqual(neutral.uci(), "e2e4")
        self.assertEqual(personal.uci(), "d2d4")
        self.assertEqual(after["prior_move"], "e4")
        self.assertTrue(after["personal_changed"])
        self.assertTrue(after["engine_changed"])
        self.assertEqual(after["sample_count"], 40)
        self.assertAlmostEqual(after["expected_regret_cp"], 160)
        self.assertAlmostEqual(after["prior_expected_regret_cp"], 20)
        self.assertEqual(after["engine_cost_cp"], 30)
        self.assertFalse(before["personal_changed"])
        self.assertEqual(board.fen(), chess.STARTING_FEN)
        self.assertEqual(board.move_stack, [])

    def test_maia_prior_change_is_not_reported_as_personal_learning(self):
        room = EngineRoom(SearchFixture(), MaiaFixture({"e2e4": .1, "d2d4": .8}))
        move, decision = room.choose(chess.Board(), PolicyFixture(samples=40))
        self.assertEqual(move.uci(), "d2d4")
        self.assertEqual(decision["baseline_move"], "e4")
        self.assertEqual(decision["prior_move"], "d4")
        self.assertTrue(decision["engine_changed"])
        self.assertFalse(decision["personal_changed"])
        self.assertAlmostEqual(decision["expected_regret_cp"], decision["prior_expected_regret_cp"])

    def test_65cp_boundary_is_included_and_66cp_candidate_never_evaluated(self):
        roots = [row("e2e4", 50), row("d2d4", 50 - MAX_COST),
                 row("c2c4", 50 - MAX_COST - 1)]
        maia = MaiaFixture()
        room = EngineRoom(SearchFixture(roots, loss=3000), maia)
        move, decision = room.choose(
            chess.Board(), PolicyFixture({"e2e4": 0, "d2d4": .6, "c2c4": 1}, samples=40)
        )
        self.assertEqual(MAX_COST, 65)
        self.assertEqual(move.uci(), "d2d4")
        self.assertEqual(decision["engine_cost_cp"], 65)
        self.assertEqual({item["move"] for item in decision["candidates"]}, {"e4", "d4"})
        self.assertNotIn("c2c4", maia.calls)
        # The reply's 3000cp loss is capped at 2000 before taking its expectation.
        self.assertEqual(decision["expected_regret_cp"], .6 * 2000)

    def test_forced_mate_band_preserves_search_choice_for_wins_and_losses(self):
        for best_score, second_score in ((MATE - 5, MATE - 10), (-MATE + 10, -MATE + 5)):
            with self.subTest(best_score=best_score):
                maia = MaiaFixture()
                room = EngineRoom(SearchFixture([row("e2e4", best_score),
                                                 row("d2d4", second_score)]), maia)
                move, decision = room.choose(
                    chess.Board(), PolicyFixture({"e2e4": 0, "d2d4": 1}, samples=40)
                )
                self.assertEqual(move.uci(), "e2e4")
                self.assertEqual(len(decision["candidates"]), 1)
                self.assertEqual(decision["engine_cost_cp"], 0)
                self.assertFalse(decision["personal_changed"])
                self.assertNotIn("d2d4", maia.calls)

    def test_baseline_mode_preserves_stockfish_choice(self):
        room = EngineRoom(SearchFixture(), MaiaFixture({"d2d4": .9}))
        move, decision = room.choose(
            chess.Board(), PolicyFixture({"d2d4": 1}, samples=40), mode="baseline"
        )
        self.assertEqual(move.uci(), "e2e4")
        self.assertFalse(decision["engine_changed"])
        self.assertFalse(decision["personal_changed"])
        self.assertEqual(len(decision["candidates"]), 1)

    def test_checkmate_and_stalemate_candidates_skip_human_policy(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        for uci, score in (("f7g7", MATE - 1), ("f7e6", 0)):
            with self.subTest(uci=uci):
                resulting = board.copy()
                resulting.push_uci(uci)
                self.assertTrue(resulting.is_game_over())
                stockfish, maia, policy = MagicMock(), MagicMock(), MagicMock()
                stockfish.name = "Stockfish fixture"
                stockfish.rank.return_value = [row(uci, score)]
                maia.info = {"name": "Maia fixture"}
                policy.samples = 0
                move, decision = EngineRoom(stockfish, maia).choose(board, policy)
                self.assertEqual(move.uci(), uci)
                self.assertEqual(decision["expected_regret_cp"], 0)
                self.assertEqual(decision["replies"], [])
                maia.probabilities.assert_not_called()
                policy.distribution.assert_not_called()
                stockfish.rank.assert_called_once()

    def test_stockfish_unavailable_fails_without_maia_or_fallback(self):
        stockfish, maia = MagicMock(), MagicMock()
        stockfish.rank.side_effect = EngineUnavailable("Stockfish is missing")
        with self.assertRaisesRegex(EngineUnavailable, "Stockfish is missing"):
            EngineRoom(stockfish, maia).choose(chess.Board(), PolicyFixture())
        maia.probabilities.assert_not_called()


class StockfishProtocolTests(TestCase):
    def test_black_root_scores_use_black_perspective_and_sort_correctly(self):
        board = chess.Board()
        board.push_san("e4")
        engine = MagicMock()
        engine.analyse.return_value = [
            {"pv": [chess.Move.from_uci("e7e5")],
             "score": chess.engine.PovScore(chess.engine.Cp(80), chess.WHITE), "depth": 12},
            {"pv": [chess.Move.from_uci("c7c5")],
             "score": chess.engine.PovScore(chess.engine.Cp(-20), chess.WHITE), "depth": 13},
        ]
        stockfish = Stockfish("/not-needed/stockfish")
        stockfish.engine = engine
        result = stockfish.rank(board, count=2, nodes=1234)
        self.assertEqual([(item["move"].uci(), item["score"]) for item in result],
                         [("c7c5", 20), ("e7e5", -80)])
        self.assertEqual(engine.analyse.call_args.args[1].nodes, 1234)
        self.assertEqual(engine.analyse.call_args.kwargs["multipv"], 2)

    def test_incomplete_duplicate_missing_score_and_illegal_uci_rows_fail(self):
        valid = {"pv": [chess.Move.from_uci("e2e4")],
                 "score": chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE)}
        cases = [[], [valid], [valid, valid],
                 [valid, {"pv": [chess.Move.from_uci("d2d4")]}],
                 [valid, {"pv": [chess.Move.from_uci("e2e5")], "score": valid["score"]}],
                 [valid, {"pv": [], "score": valid["score"]}]]
        for analysis in cases:
            with self.subTest(analysis=analysis):
                stockfish = Stockfish("/not-needed/stockfish")
                stockfish.engine = MagicMock()
                stockfish.engine.analyse.return_value = analysis
                with self.assertRaises(EngineUnavailable):
                    stockfish.rank(chess.Board(), count=2)

    def test_missing_binary_has_no_search_fallback(self):
        with patch("chess_tutor.opponent.Path.is_file", return_value=False), \
                patch("chess.engine.SimpleEngine.popen_uci") as popen:
            with self.assertRaisesRegex(EngineUnavailable, "Stockfish is missing"):
                Stockfish("/missing/stockfish").rank(chess.Board())
            popen.assert_not_called()

    def test_engine_crash_closes_process_and_raises(self):
        stockfish = Stockfish("/not-needed/stockfish")
        engine = MagicMock()
        engine.analyse.side_effect = chess.engine.EngineTerminatedError("crashed")
        stockfish.engine = engine
        with self.assertRaisesRegex(EngineUnavailable, "analysis failed"):
            stockfish.rank(chess.Board())
        engine.quit.assert_called_once()
        self.assertIsNone(stockfish.engine)

    def test_nodes_only_search_has_an_external_deadline(self):
        stockfish = Stockfish('/not-needed/stockfish')
        stockfish.timeout = .02
        engine = MagicMock()
        closed = threading.Event()
        engine.close.side_effect = closed.set

        def hang(*args, **kwargs):
            self.assertIsNone(args[1].time)
            self.assertTrue(closed.wait(1), 'Watchdog did not close the hung engine')
            raise chess.engine.EngineTerminatedError('Closed by watchdog')

        engine.analyse.side_effect = hang
        stockfish.engine = engine
        with self.assertRaisesRegex(EngineUnavailable, 'timed out after 0.02 seconds'):
            stockfish.rank(chess.Board())
        self.assertTrue(closed.is_set())
        self.assertIsNone(stockfish.engine)
        engine.quit.assert_not_called()


@skipUnless(os.environ.get("NEMESIS_TEST_REAL_ENGINES") == "1", "Set NEMESIS_TEST_REAL_ENGINES=1")
class RealEngineRoomTests(TestCase):
    def test_real_adaptive_decision_is_legal_and_within_cost_limit(self):
        from .player_policy import PersonalPolicy

        room = EngineRoom()
        self.addCleanup(room.close)
        status = room.status()
        if not status["ready"]:
            self.skipTest(status["engine_error"])
        board = chess.Board()
        board.push_san("e4")
        move, decision = room.choose(board, PersonalPolicy())
        self.assertIn(move, board.legal_moves)
        self.assertLessEqual(decision["engine_cost_cp"], MAX_COST)
        self.assertFalse(decision["personal_changed"])
        self.assertGreater(decision["root_nodes"], 0)
        self.assertTrue(decision["replies"])
        self.assertAlmostEqual(sum(reply["prior_probability"] for reply in decision["replies"]), 1)
        self.assertAlmostEqual(sum(reply["personal_probability"] for reply in decision["replies"]), 1)
