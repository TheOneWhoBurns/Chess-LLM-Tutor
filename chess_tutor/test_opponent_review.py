"""Safety regressions when reply search finds a refutation missed at the root."""
from unittest import TestCase
from unittest.mock import MagicMock

import chess

from .opponent import EngineRoom, MATE


class ReplyRefutationTests(TestCase):
    def choose_with_refutation(self, reply_score):
        board = chess.Board("6k1/8/3b4/8/7q/8/PP3PPP/5RK1 w - - 0 1")

        def row(move, score, mate=None):
            return {"move": chess.Move.from_uci(move), "score": score, "mate": mate,
                    "depth": 12, "nodes": 1000}

        def rank(position, count=6, nodes=None):
            if not position.move_stack:
                # The initial search underestimates the threat against a3.
                return [row("h2h3", 50), row("a2a3", 20)]
            responses = []
            for move in position.legal_moves:
                refutation = position.peek().uci() == "a2a3" and move.uci() == "h4h2"
                responses.append(row(move.uci(), reply_score if refutation else 0,
                                     1 if refutation and reply_score >= MATE - 1000 else None))
            return sorted(responses, key=lambda item: -item["score"])

        stockfish, maia, policy = MagicMock(), MagicMock(), MagicMock()
        stockfish.name = "Stockfish search fixture"
        stockfish.rank.side_effect = rank
        maia.info = {"name": "Maia fixture"}
        maia.probabilities.side_effect = lambda position: {
            move.uci(): 1 / position.legal_moves.count() for move in position.legal_moves}
        policy.samples = 100
        policy.distribution.side_effect = lambda position, prior: prior
        return board, EngineRoom(stockfish, maia).choose(board, policy)

    def test_reply_search_discovered_mate_cannot_be_traded_for_expected_mistakes(self):
        board, (move, decision) = self.choose_with_refutation(MATE - 1)
        refutation = board.copy()
        refutation.push_uci("a2a3")
        refutation.push_uci("h4h2")
        self.assertTrue(refutation.is_checkmate())
        self.assertEqual(move.uci(), "h2h3")
        self.assertLessEqual(decision["engine_cost_cp"], 65)
        rejected = next(candidate for candidate in decision['candidates'] if candidate['move'] == 'a3')
        self.assertFalse(rejected['eligible'])
        self.assertEqual(rejected['reply_score_cp'], -MATE + 1)
        self.assertEqual(rejected['guard_reason'], 'reply_search_found_losing_mate')

    def test_reply_search_discovered_cost_must_recheck_the_65cp_guard(self):
        _, (move, decision) = self.choose_with_refutation(400)
        # Reply search now evaluates a3 at -400 from the root perspective,
        # while h3 is 0. The earlier +20 vs +50 estimate is no longer enough.
        self.assertEqual(move.uci(), "h2h3")
        self.assertLessEqual(decision["engine_cost_cp"], 65)
        rejected = next(candidate for candidate in decision['candidates'] if candidate['move'] == 'a3')
        self.assertEqual(rejected['engine_cost_cp'], 30)
        self.assertEqual(rejected['reply_cost_cp'], 400)
        self.assertFalse(rejected['eligible'])
        self.assertEqual(rejected['guard_reason'], 'reply_cost_exceeds_limit')

    def test_refreshed_values_drive_objective_without_claiming_personal_learning(self):
        board = chess.Board()
        stockfish, maia, policy = MagicMock(), MagicMock(), MagicMock()
        stockfish.name = 'Stockfish fixture'

        def rank(position, count=6, nodes=None):
            if not position.move_stack:
                pairs = [('e2e4', 50), ('d2d4', 20)]
            else:
                # Later search values d4 +40 and e4 0 from the root POV;
                # every human response has the same value, so regret is zero.
                score = -40 if position.peek().uci() == 'd2d4' else 0
                pairs = [(move.uci(), score) for move in position.legal_moves]
            return [{'move': chess.Move.from_uci(uci), 'score': score,
                     'mate': None, 'depth': 12, 'nodes': 1000} for uci, score in pairs]

        stockfish.rank.side_effect = rank
        maia.info = {'name': 'Maia fixture'}
        maia.probabilities.side_effect = lambda position: {
            move.uci(): 1 / position.legal_moves.count() for move in position.legal_moves}
        policy.samples = 100
        policy.distribution.side_effect = lambda position, prior: prior
        move, decision = EngineRoom(stockfish, maia).choose(board, policy)
        self.assertEqual(move.uci(), 'd2d4')
        self.assertEqual(decision['baseline_move'], 'e4')
        self.assertEqual(decision['reply_baseline_move'], 'd4')
        self.assertEqual(decision['reply_score_cp'], 40)
        self.assertEqual(decision['reply_cost_cp'], 0)
        self.assertEqual(decision['expected_regret_cp'], 0)
        self.assertFalse(decision['personal_changed'])
