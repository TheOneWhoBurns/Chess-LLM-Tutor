"""Online opponent modelling and bounded, position-aware chess move selection.

The small MLP learns estimated move regret, not chess rules or a grandmaster policy.
All scores are centipawns from the side-to-move at the root of a search.
"""
import logging
import os
from contextlib import AbstractContextManager

import chess
import chess.engine
import numpy as np

LOGGER = logging.getLogger(__name__)
VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
          chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
CONTEXTS = ("Tactical pressure", "King safety", "Development", "Pawn structure", "Endgames")
THRESHOLDS = (.08, .18, .4, .12, .55)
INPUTS, HIDDEN = 12, 16
MATE = 100000


def features(board, player=chess.WHITE):
    """Bounded features of the position BEFORE the human's decision."""
    own = board.occupied_co[player]
    phase = min(1., sum(len(board.pieces(p, c)) * v for p, v in
                       ((chess.KNIGHT, 1), (chess.BISHOP, 1), (chess.ROOK, 2), (chess.QUEEN, 4))
                       for c in chess.COLORS) / 24)
    attacked = sum(VALUES[board.piece_type_at(s)] for s in chess.scan_forward(own)
                   if board.piece_type_at(s) != chess.KING and board.is_attacked_by(not player, s))
    loose = sum(VALUES[board.piece_type_at(s)] for s in chess.scan_forward(own)
                if board.is_attacked_by(not player, s) and not board.is_attacked_by(player, s))
    king = board.king(player)
    ring = list(chess.SquareSet(chess.BB_KING_ATTACKS[king])) if king is not None else []
    pressure = sum(board.is_attacked_by(not player, s) for s in ring) / 8
    if king is not None and chess.square_file(king) in (3, 4) and board.fullmove_number > 5:
        pressure += .25 * phase
    back_rank = 0 if player else 7
    undeveloped = sum(board.piece_at(chess.square(f, back_rank)) == chess.Piece(p, player)
                      for f, p in ((1, chess.KNIGHT), (2, chess.BISHOP), (5, chess.BISHOP), (6, chess.KNIGHT)))
    pawns = board.pieces(chess.PAWN, player)
    files = [sum(chess.square_file(s) == f for s in pawns) for f in range(8)]
    doubled = sum(max(0, n - 1) for n in files)
    isolated = sum(n for f, n in enumerate(files)
                   if (f == 0 or not files[f - 1]) and (f == 7 or not files[f + 1]))
    material = sum(VALUES[p] * (len(board.pieces(p, player)) - len(board.pieces(p, not player)))
                   for p in VALUES)
    return np.clip(np.array([
        attacked / 1500, pressure, undeveloped / 4 if board.fullmove_number <= 15 else 0,
        (doubled + isolated) / 8, 1 - phase, material / 4000,
        loose / 1000, phase, len(pawns) / 8, board.legal_moves.count() / 60,
        float(board.is_check()), min(board.fullmove_number / 60, 1)
    ], dtype=float), -1, 1)


class PlayerNetwork:
    """12 → 16 tanh → 1 sigmoid MLP, updated by replay-based gradient descent."""
    def __init__(self, saved=None):
        if saved:
            self.w1 = np.array(saved["w1"], dtype=float)
            self.b1 = np.array(saved["b1"], dtype=float)
            self.w2 = np.array(saved["w2"], dtype=float)
            self.b2 = float(saved["b2"])
            self.samples = int(saved["samples"])
            self.replay = saved["replay"]
        else:
            rng = np.random.default_rng(17)
            self.w1 = rng.normal(0, .22, (INPUTS, HIDDEN))
            self.b1 = np.zeros(HIDDEN)
            self.w2 = rng.normal(0, .15, HIDDEN)
            self.b2 = -1.5
            self.samples, self.replay = 0, []

    @property
    def influence(self):
        return 0. if self.samples < 8 else min(1., self.samples / 40)

    def predict(self, x):
        hidden = np.tanh(np.asarray(x) @ self.w1 + self.b1)
        return 1 / (1 + np.exp(-np.clip(hidden @ self.w2 + self.b2, -30, 30)))

    def learn(self, x, loss_cp):
        target = float(np.clip(loss_cp / 300, 0, 1))
        self.replay = (self.replay + [[np.asarray(x).tolist(), target]])[-128:]
        self.samples += 1
        x_batch = np.array([item[0] for item in self.replay])
        y_batch = np.array([item[1] for item in self.replay])
        # Full bounded replay reduces forgetting; BCE gradients permit soft regret labels.
        for _ in range(24):
            hidden = np.tanh(x_batch @ self.w1 + self.b1)
            prediction = 1 / (1 + np.exp(-np.clip(hidden @ self.w2 + self.b2, -30, 30)))
            delta = (prediction - y_batch) / len(y_batch)
            hidden_delta = np.outer(delta, self.w2) * (1 - hidden ** 2)
            self.w2 -= .12 * (hidden.T @ delta + .0005 * self.w2)
            self.b2 -= .12 * float(delta.sum())
            self.w1 -= .12 * (x_batch.T @ hidden_delta + .0005 * self.w1)
            self.b1 -= .12 * hidden_delta.sum(axis=0)

    def dump(self):
        return {"version": 1, "w1": self.w1.tolist(), "b1": self.b1.tolist(),
                "w2": self.w2.tolist(), "b2": self.b2, "samples": self.samples,
                "replay": self.replay}


def static_score(board):
    """Lightweight white-relative evaluation used by the self-contained demo search."""
    if board.is_checkmate():
        return -MATE if board.turn else MATE
    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0
    score = 0
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square) if piece.color else 7 - chess.square_rank(square)
        file = chess.square_file(square)
        center = 3.5 - (abs(file - 3.5) + abs(rank - 3.5)) / 2
        value = VALUES[piece.piece_type]
        if piece.piece_type == chess.PAWN:
            value += rank * 7 + center * 3
        elif piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            value += center * 15 + (12 if rank > 0 else -12)
        elif piece.piece_type == chess.KING:
            value += 25 if rank == 0 and file in (2, 6) else 0
        score += value if piece.color else -value
    return score


def demo_candidates(board):
    """Two-ply minimax: every legal move followed by the opponent's best reply."""
    sign = 1 if board.turn else -1
    candidates = []
    for move in list(board.legal_moves):
        board.push(move)
        if board.is_game_over():
            value = sign * static_score(board)
        else:
            value = float("inf")
            for reply in list(board.legal_moves):
                board.push(reply)
                value = min(value, sign * static_score(board))
                board.pop()
        board.pop()
        candidates.append({"move": move, "score": value})
    return sorted(candidates, key=lambda c: (-c["score"], c["move"].uci()))


class Evaluator(AbstractContextManager):
    """Optional UCI backend; demo search needs no binary, weights, or API key."""
    def __init__(self):
        self.engine = None
        self.name = "Demo search · 2 ply"

    def __enter__(self):
        path = os.environ.get("NEMESIS_UCI_ENGINE")
        if path:
            command = [path]
            if os.environ.get("NEMESIS_UCI_WEIGHTS"):
                command.append("--weights=" + os.environ["NEMESIS_UCI_WEIGHTS"])
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(command, timeout=10)
                self.name = self.engine.id.get("name", "UCI engine")
            except (OSError, chess.engine.EngineError, TimeoutError):
                LOGGER.warning("UCI engine unavailable; using demo search.")
                self.name = "Demo search · 2 ply (UCI unavailable)"
        return self

    def rank(self, board):
        if board.is_game_over():
            return []
        if self.engine:
            try:
                rows = self.engine.analyse(board, chess.engine.Limit(time=.4),
                                           multipv=board.legal_moves.count())
                candidates = [{"move": row["pv"][0],
                               "score": row["score"].pov(board.turn).score(mate_score=MATE)}
                              for row in rows if row.get("pv") and row.get("score") is not None]
                if len({row["move"] for row in candidates}) == board.legal_moves.count():
                    self.name = self.engine.id.get("name", "UCI engine")
                    return sorted(candidates, key=lambda c: (-c["score"], c["move"].uci()))
            except (chess.engine.EngineError, TimeoutError):
                LOGGER.warning("UCI analysis failed; using demo search.")
            self.name = "Demo search · 2 ply (UCI fallback)"
        return demo_candidates(board)

    def __exit__(self, *args):
        if self.engine:
            try:
                self.engine.quit()
            except (chess.engine.EngineError, TimeoutError):
                self.engine.close()


def choose_move(board, candidates, network, adaptive=True):
    """Rerank within 65cp of the best search result. Never trade a forced result."""
    best = candidates[0]
    if abs(best["score"]) >= MATE - 1000:
        return best, False
    pool = [c for c in candidates if c["score"] >= best["score"] - 65]
    for candidate in pool:
        board.push(candidate["move"])
        risk = 0. if board.is_game_over() else float(network.predict(features(board, board.turn)))
        board.pop()
        candidate["personal_score"] = candidate["score"] + (
            240 * network.influence * risk if adaptive else 0)
    choice = max(pool, key=lambda c: c["personal_score"])
    return choice, choice["move"] != best["move"]


def observe(stats, x, loss_cp):
    for index, name in enumerate(CONTEXTS):
        row = stats.setdefault(name, {"positions": 0, "errors": 0, "loss_cp": 0})
        if x[index] >= THRESHOLDS[index]:
            row["positions"] += 1
            row["errors"] += int(loss_cp >= 100)
            row["loss_cp"] += round(min(1000, loss_cp))


def weakness_summary(stats):
    rows = []
    for name in CONTEXTS:
        item = stats.get(name, {"positions": 0, "errors": 0, "loss_cp": 0})
        n = item["positions"]
        rows.append({"name": name, **item, "rate": round(100 * item["errors"] / n) if n else 0,
                     "supported": n >= 5, "mean_loss": round(item["loss_cp"] / n) if n else 0})
    return rows
