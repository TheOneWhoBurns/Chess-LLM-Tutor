"""Policy protocol tests; opt into the downloaded neural network smoke test.

NEMESIS_TEST_REAL_MAIA=1 python -m unittest chess_tutor.test_maia_policy -v
"""

import math
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import chess
import chess.engine

from .maia_policy import MaiaPolicy, MaiaUnavailable, POLICY_FLOOR, parse_policy_lines


CASTLING_FEN = "r3k2r/ppp2ppp/2n1bn2/3p4/3P4/2N1BN2/PPP2PPP/R3K2R w KQkq - 0 10"


def policy_lines(board, weights=None):
    moves = list(board.legal_moves)
    weights = weights or {move.uci(): 100 / len(moves) for move in moves}
    lines = []
    for move in moves:
        raw = board.uci(move, chess960=True)
        lines.append(f"{raw} ( 12) N: 0 (+ 0) (P: {weights[move.uci()]:.2f}%) (Q: 0)")
    return lines


class PolicyParserTests(unittest.TestCase):
    def test_full_normalized_policy_and_nonmove_summary(self):
        board = chess.Board()
        lines = policy_lines(board) + ["node (20) N: 1 (P: 0.00%) (Q: 0)"]
        result = parse_policy_lines(board, lines)
        self.assertEqual(set(result), {move.uci() for move in board.legal_moves})
        self.assertAlmostEqual(sum(result.values()), 1)
        self.assertTrue(all(value == 0.05 for value in result.values()))

    def test_missing_policy_move_is_an_error_not_uniform_fallback(self):
        with self.assertRaisesRegex(MaiaUnavailable, "19/20"):
            parse_policy_lines(chess.Board(), policy_lines(chess.Board())[:-1])

    def test_rounded_zeros_keep_positive_mass(self):
        board = chess.Board()
        weights = {move.uci(): 0 for move in board.legal_moves}
        weights["e2e4"] = 100
        result = parse_policy_lines(board, policy_lines(board, weights))
        self.assertTrue(all(math.isfinite(value) and value > 0 for value in result.values()))
        self.assertGreater(result["e2e4"], 0.999)
        self.assertAlmostEqual(result["a2a3"], POLICY_FLOOR / (1 + 19 * POLICY_FLOOR))
        self.assertGreater(result["a2a3"], .00002)
        self.assertAlmostEqual(sum(result.values()), 1)

    def test_illegal_policy_move_is_rejected(self):
        lines = policy_lines(chess.Board()) + ["e2e5 (12) N: 0 (P: 0.00%)"]
        with self.assertRaisesRegex(MaiaUnavailable, "illegal policy move"):
            parse_policy_lines(chess.Board(), lines)

    def test_complete_but_corrupt_mass_is_rejected(self):
        board = chess.Board()
        weights = {move.uci(): 1 for move in board.legal_moves}
        with self.assertRaisesRegex(MaiaUnavailable, "total mass"):
            parse_policy_lines(board, policy_lines(board, weights))

    def test_castling_king_to_rook_is_normalized_for_both_colors(self):
        for color in (chess.WHITE, chess.BLACK):
            board = chess.Board(CASTLING_FEN)
            board.turn = color
            result = parse_policy_lines(board, policy_lines(board))
            rank = "1" if color else "8"
            self.assertIn(f"e{rank}g{rank}", result)
            self.assertIn(f"e{rank}c{rank}", result)
            self.assertNotIn(f"e{rank}h{rank}", result)
            self.assertNotIn(f"e{rank}a{rank}", result)
            self.assertAlmostEqual(sum(result.values()), 1)

    def test_promotion_suffixes_are_retained(self):
        board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
        result = parse_policy_lines(board, policy_lines(board))
        for piece in "qrbn":
            self.assertIn("a7a8" + piece, result)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.weights = Path(temp.name) / "maia-1500.pb.gz"
        self.weights.touch()

    def test_lazy_start_and_missing_weights_report_actionable_error(self):
        with patch("chess.engine.SimpleEngine.popen_uci") as start:
            policy = MaiaPolicy(self.weights.parent / "missing.pb.gz", "/test/lc0")
            self.assertFalse(policy.info["ready"])
            start.assert_not_called()
            with self.assertRaisesRegex(MaiaUnavailable, "weights are missing"):
                policy.probabilities(chess.Board())
            start.assert_not_called()

    def test_streams_all_info_strings_and_resets_tree(self):
        board = chess.Board()
        engine = MagicMock()
        engine.id = {"name": "Lc0 test"}
        stream = engine.analysis.return_value.__enter__.return_value
        stream.__iter__.side_effect = lambda: iter(
            [{"string": line} for line in policy_lines(board)]
        )
        with patch("chess.engine.SimpleEngine.popen_uci", return_value=engine) as start:
            policy = MaiaPolicy(self.weights, "/test/lc0")
            self.assertEqual(len(policy.probabilities(board)), 20)
            self.assertEqual(len(policy.probabilities(board)), 20)
            start.assert_called_once()
            first, second = engine.analysis.call_args_list
            self.assertIsNot(first.kwargs["game"], second.kwargs["game"])
            self.assertEqual(first.args[1].nodes, 1)
            self.assertIn("--policy-softmax-temp=1", start.call_args.args[0])
            self.assertEqual(policy.info["version"], "Lc0 test")
            policy.close()
            engine.close.assert_called_once()
            self.assertFalse(policy.info["ready"])

    def test_incomplete_output_closes_the_child(self):
        engine = MagicMock()
        engine.id = {"name": "Lc0 test"}
        engine.analysis.return_value.__enter__.return_value.__iter__.return_value = iter([])
        with patch("chess.engine.SimpleEngine.popen_uci", return_value=engine):
            policy = MaiaPolicy(self.weights, "/test/lc0")
            with self.assertRaisesRegex(MaiaUnavailable, "incomplete"):
                policy.probabilities(chess.Board())
            engine.close.assert_called_once()
            self.assertFalse(policy.info["ready"])

    def test_hung_stream_is_closed_at_external_deadline(self):
        closed = threading.Event()
        engine = MagicMock()
        engine.id = {"name": "Hung engine"}
        engine.close.side_effect = closed.set

        def wait_for_watchdog():
            self.assertTrue(closed.wait(1), "Watchdog did not close the process")
            raise chess.engine.EngineTerminatedError("closed")
            yield  # This is the blocking analysis iterator.

        stream = engine.analysis.return_value.__enter__.return_value
        stream.__iter__.side_effect = wait_for_watchdog
        with patch("chess.engine.SimpleEngine.popen_uci", return_value=engine):
            policy = MaiaPolicy(self.weights, "/test/lc0")
            policy.timeout = 0.02
            with self.assertRaisesRegex(MaiaUnavailable, "timed out"):
                policy.probabilities(chess.Board())
            self.assertTrue(closed.is_set())
            self.assertFalse(policy.info["ready"])


@unittest.skipUnless(os.environ.get("NEMESIS_TEST_REAL_MAIA") == "1", "Set NEMESIS_TEST_REAL_MAIA=1")
class RealMaiaTests(unittest.TestCase):
    def test_start_castling_promotion_and_black_probabilities(self):
        boards = [chess.Board(), chess.Board(CASTLING_FEN),
                  chess.Board(CASTLING_FEN.replace(" w ", " b ")),
                  chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")]
        with MaiaPolicy() as policy:
            for board in boards:
                with self.subTest(fen=board.fen()):
                    before = board.fen()
                    result = policy.probabilities(board)
                    self.assertEqual(set(result), {move.uci() for move in board.legal_moves})
                    self.assertAlmostEqual(sum(result.values()), 1)
                    self.assertTrue(all(math.isfinite(p) and p > 0 for p in result.values()))
                    self.assertEqual(board.fen(), before)
            opening = policy.probabilities(chess.Board())
            self.assertGreater(max(opening.values()), 0.2)


if __name__ == "__main__":
    unittest.main()
