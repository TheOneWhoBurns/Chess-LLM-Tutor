"""A personal legal-move policy learned from the player's actual choices.

The caller supplies a frozen human-play prior (Maia in the application). A small
neural residual changes its probabilities using demonstrated moves, never engine
regret or guessed weakness labels. New profiles reproduce the supplied prior.

The sample-dependent adaptation weight is a conservative scheduling parameter,
not confidence. Neither its value nor a lower training loss demonstrates that
personalization predicts a player's future moves; use the pre-update diagnostics.
"""
from collections import OrderedDict
import math
from threading import Lock

import chess
import numpy as np


SCHEMA_VERSION = 2
FEATURE_VERSION = 1
INPUTS = 222
HIDDEN = 24
MAX_REPLAY = 96
MAX_BATCH = 24
MAX_FIT_BATCH = 64
MAX_FEATURE_CACHE = 512
EPSILON = 1e-12
MAX_RESIDUAL = 2.0
PRIOR_PENALTY = 0.08
WEIGHT_PENALTY = 0.0005
LEARNING_RATE = 0.12
TRAINING_STEPS = 10


_FEATURE_CACHE = OrderedDict()
_FEATURE_CACHE_LOCK = Lock()


def _normalise_prior(board, prior):
    if not isinstance(board, chess.Board) or not board.is_valid():
        raise ValueError("The player policy requires a valid chess position.")
    moves = sorted(move.uci() for move in board.legal_moves)
    if not isinstance(prior, dict) or set(prior) != set(moves):
        raise ValueError("The prior must contain exactly every legal UCI move.")
    if not moves:
        return moves, np.empty(0, dtype=float)
    values = []
    for move in moves:
        value = prior[move]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError("Prior probabilities must be finite, nonnegative numbers.")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("Prior probabilities must be finite, nonnegative numbers.")
        values.append(number)
    probabilities = np.asarray(values, dtype=float)
    largest = float(probabilities.max())
    if largest <= 0:
        raise ValueError("The prior must assign some probability to a legal move.")
    # Scaling first also handles large, finite unnormalised weights without overflow.
    probabilities /= largest
    probabilities /= probabilities.sum()
    probabilities = np.maximum(probabilities, EPSILON)
    probabilities /= probabilities.sum()
    return moves, probabilities


def _feature_matrix(board, moves):
    """Reuse immutable candidate features across replay and restored profiles.

    Full FEN retains the move counters used by the features. Repetition history
    is deliberately absent: no feature depends on it. Keep the move order in the
    key so each matrix row always matches its corresponding prior probability.
    """
    key = (board.fen(en_passant="fen"), tuple(moves))
    with _FEATURE_CACHE_LOCK:
        if key in _FEATURE_CACHE:
            _FEATURE_CACHE.move_to_end(key)
            return _FEATURE_CACHE[key]
    matrix = _compute_feature_matrix(board, moves)
    # Bytes-backed storage cannot be made writable by a caller. Training only
    # reads these inputs and concatenates its own batch matrix.
    matrix = np.frombuffer(matrix.tobytes(), dtype=matrix.dtype).reshape(matrix.shape)
    with _FEATURE_CACHE_LOCK:
        _FEATURE_CACHE[key] = matrix
        _FEATURE_CACHE.move_to_end(key)
        if len(_FEATURE_CACHE) > MAX_FEATURE_CACHE:
            _FEATURE_CACHE.popitem(last=False)
    return matrix


def _compute_feature_matrix(board, moves):
    """Side-relative board occupancy, exact from/to squares and move properties.

    These are observed position/move attributes, not chess evaluation scores.
    A private copy is used for checking-move detection so the caller's board,
    including its repetition history, is never modified.
    """
    player = board.turn
    orient = (lambda square: square) if player else chess.square_mirror
    position = np.zeros(64, dtype=float)
    for square, piece in board.piece_map().items():
        position[orient(square)] = piece.piece_type / 6 * (1 if piece.color == player else -1)
    counts = [len(board.pieces(kind, color)) for color in (player, not player)
              for kind in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)]
    phase = min(sum(count * weight for count, weight in
                    zip(counts, (1, 1, 2, 4, 1, 1, 2, 4))) / 24, 1.0)
    king = board.king(player)
    context = np.array([
        phase, min(board.fullmove_number / 80, 1), min(board.halfmove_clock / 100, 1),
        float(board.is_check()), min(len(moves) / 80, 1),
        len(board.pieces(chess.PAWN, player)) / 8,
        len(board.pieces(chess.PAWN, not player)) / 8,
        float(board.has_kingside_castling_rights(player)),
        float(board.has_queenside_castling_rights(player)),
        chess.square_rank(orient(king)) / 7,
    ], dtype=float)
    matrix = np.zeros((len(moves), INPUTS), dtype=float)
    matrix[:, :64] = position
    matrix[:, 212:] = context
    scratch = board.copy(stack=False)
    for index, uci in enumerate(moves):
        move = chess.Move.from_uci(uci)
        matrix[index, 64 + orient(move.from_square)] = 1
        matrix[index, 128 + orient(move.to_square)] = 1
        matrix[index, 192 + board.piece_type_at(move.from_square) - 1] = 1
        captured = board.piece_type_at(move.to_square)
        if board.is_en_passant(move):
            captured = chess.PAWN
        if captured:
            matrix[index, 198 + captured - 1] = 1
        if move.promotion:
            matrix[index, 204 + move.promotion - chess.KNIGHT] = 1
        matrix[index, 208:212] = (
            board.is_capture(move), scratch.gives_check(move),
            board.is_castling(move), board.is_en_passant(move),
        )
    return matrix


def _softmax(logits):
    weights = np.exp(logits - np.max(logits))
    return weights / weights.sum()


def _saved_array(saved, name, shape):
    try:
        value = np.asarray(saved[name], dtype=float)
    except (KeyError, ValueError, TypeError, OverflowError) as error:
        raise ValueError("Invalid saved player-policy weights.") from error
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Invalid saved player-policy weights.")
    return value.copy()


class PersonalPolicy:
    """222 → 24 tanh → bounded scalar residual, normalized over legal moves."""

    def __init__(self, saved=None):
        rng = np.random.default_rng(1701)
        self.w1 = rng.normal(0, 0.1, (INPUTS, HIDDEN))
        self.b1 = np.zeros(HIDDEN)
        # A zero output head ensures no invented preferences at initialization.
        self.w2 = np.zeros(HIDDEN)
        self.b2 = 0.0
        self.samples = 0
        self.replay = []
        if saved is not None:
            self._restore(saved)

    @property
    def adaptation_weight(self):
        """Sample-count schedule, NOT a measure of prediction confidence."""
        return self.samples / (self.samples + 64)

    def _restore(self, saved):
        if not isinstance(saved, dict) or saved.get("version") != SCHEMA_VERSION or \
                saved.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Unsupported saved player policy; expected schema version 2.")
        samples = saved.get("samples")
        replay = saved.get("replay")
        if type(samples) is not int or samples < 0 or not isinstance(replay, list) or \
                len(replay) != min(samples, MAX_REPLAY):
            raise ValueError("Invalid saved player-policy observations.")
        self.w1 = _saved_array(saved, "w1", (INPUTS, HIDDEN))
        self.b1 = _saved_array(saved, "b1", (HIDDEN,))
        self.w2 = _saved_array(saved, "w2", (HIDDEN,))
        self.b2 = float(_saved_array(saved, "b2", ()))
        restored = []
        for row in replay:
            try:
                if not isinstance(row, dict) or not isinstance(row.get("fen"), str):
                    raise ValueError("Invalid replay position.")
                board = chess.Board(row["fen"])
                moves, probabilities = _normalise_prior(board, row["prior"])
                if not isinstance(row["chosen"], str) or row["chosen"] not in moves:
                    raise ValueError("Invalid replay move.")
                restored.append({"fen": board.fen(), "prior": dict(zip(moves, probabilities.tolist())),
                                 "chosen": row["chosen"]})
            except (KeyError, ValueError, TypeError, IndexError) as error:
                raise ValueError("Invalid saved player-policy replay.") from error
        self.samples = samples
        self.replay = restored

    def _probabilities(self, matrix, prior):
        hidden = np.tanh(matrix @ self.w1 + self.b1)
        residual = MAX_RESIDUAL * np.tanh(hidden @ self.w2 + self.b2)
        return _softmax(np.log(prior) + self.adaptation_weight * residual)

    def distribution(self, board, prior):
        moves, probabilities = _normalise_prior(board, prior)
        if moves and self.samples:
            probabilities = self._probabilities(_feature_matrix(board, moves), probabilities)
        return dict(zip(moves, probabilities.tolist()))

    def learn(self, board, prior, chosen_uci):
        """Record a demonstrated choice; diagnostics are computed before training.

        ``samples_before`` identifies the predicting model and ``samples`` is the
        updated observation count. Log losses use natural logarithms. The top
        prediction is also from the pre-update personal distribution.
        """
        moves, probabilities = _normalise_prior(board, prior)
        if not isinstance(chosen_uci, str) or chosen_uci not in moves:
            raise ValueError("The demonstrated move must be legal in the supplied position.")
        personal = probabilities if not self.samples else self._probabilities(
            _feature_matrix(board, moves), probabilities)
        chosen_index = moves.index(chosen_uci)
        prior_probability = float(probabilities[chosen_index])
        personal_probability = float(personal[chosen_index])
        diagnostics = {
            "prior_probability": prior_probability,
            "personal_probability": personal_probability,
            "prior_log_loss": -math.log(max(prior_probability, EPSILON)),
            "personal_log_loss": -math.log(max(personal_probability, EPSILON)),
            "top_prediction": moves[int(np.argmax(personal))],
            "samples_before": self.samples,
            "samples": self.samples + 1,
        }
        self.replay = (self.replay + [{
            "fen": board.fen(), "prior": dict(zip(moves, probabilities.tolist())),
            "chosen": chosen_uci,
        }])[-MAX_REPLAY:]
        self.samples += 1
        self._train()
        return diagnostics

    @staticmethod
    def _example_rows(examples, limit):
        """Validate and detach demonstrated choices before changing a model."""
        if not isinstance(examples, list) or len(examples) > limit:
            raise ValueError(f"Expected a list of at most {limit} player-policy examples.")
        rows = []
        for example in examples:
            if not isinstance(example, dict) or not {"board", "prior", "chosen_uci"} <= example.keys():
                raise ValueError("Each example requires board, prior, and chosen_uci.")
            board = example["board"]
            moves, probabilities = _normalise_prior(board, example["prior"])
            chosen = example["chosen_uci"]
            if not isinstance(chosen, str) or chosen not in moves:
                raise ValueError("The demonstrated move must be legal in the supplied position.")
            rows.append({"fen": board.fen(), "prior": dict(zip(moves, probabilities.tolist())),
                         "chosen": chosen})
        return rows

    def fit_batch(self, examples, *, count_observations=True, steps=1):
        """Fit at most 64 supplied demonstrations without sampling replay.

        Offline callers count observations during their first epoch only. Later
        epochs use ``count_observations=False`` so repeated optimization does not
        inflate the sample-count schedule or replace the retained replay. This
        uses the online learner's objective and optimizer; it does not produce
        pre-update evaluation diagnostics. Evaluate held-out games separately.
        """
        if type(count_observations) is not bool:
            raise ValueError("count_observations must be a boolean.")
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer.")
        rows = self._example_rows(examples, MAX_FIT_BATCH)
        if not rows:
            raise ValueError("At least one player-policy example is required.")
        # Build the complete batch before mutating count, replay, or weights.
        prepared = self._prepare_rows(rows)
        samples_before = self.samples
        if count_observations:
            self.samples += len(rows)
            self.replay = (self.replay + rows)[-MAX_REPLAY:]
        self._optimise(*prepared, steps=steps)
        return {"samples_before": samples_before, "samples": self.samples,
                "examples": len(rows), "steps": steps}

    def replace_replay(self, examples):
        """Restore chronological replay after shuffled offline fitting.

        Supply the most recent ``min(samples, 96)`` counted demonstrations in
        chronological order. This changes neither weights nor sample count.
        """
        rows = self._example_rows(examples, MAX_REPLAY)
        if len(rows) != min(self.samples, MAX_REPLAY):
            raise ValueError("Replay must contain exactly the most recent min(samples, 96) examples.")
        # Also validate canonical replay positions as they will be restored.
        for row in rows:
            _normalise_prior(chess.Board(row["fen"]), row["prior"])
        self.replay = rows

    def _train(self):
        # Deterministic bounded batches include the newest observation and rotate
        # older examples; replay does not grow without limit or require an engine.
        rows = self.replay
        if len(rows) > MAX_BATCH:
            rng = np.random.default_rng(self.samples)
            selected = sorted(rng.choice(len(rows) - 1, MAX_BATCH - 1, replace=False).tolist())
            rows = [rows[index] for index in selected] + rows[-1:]
        self._optimise(*self._prepare_rows(rows), steps=TRAINING_STEPS)

    @staticmethod
    def _prepare_rows(rows):
        matrices, priors, chosen, offsets = [], [], [], [0]
        for row in rows:
            board = chess.Board(row["fen"])
            moves, probabilities = _normalise_prior(board, row["prior"])
            matrices.append(_feature_matrix(board, moves))
            priors.append(probabilities)
            chosen.append(moves.index(row["chosen"]))
            offsets.append(offsets[-1] + len(moves))
        matrix = np.concatenate(matrices)
        log_prior = np.log(np.concatenate(priors))
        return matrix, log_prior, chosen, offsets

    def _optimise(self, matrix, log_prior, chosen, offsets, *, steps):
        for _ in range(steps):
            hidden = np.tanh(matrix @ self.w1 + self.b1)
            squashed = np.tanh(hidden @ self.w2 + self.b2)
            logits = log_prior + self.adaptation_weight * MAX_RESIDUAL * squashed
            delta = np.empty(len(logits))
            for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
                prediction = _softmax(logits[start:end])
                # KL(personal || prior) discourages unsupported departures from Maia.
                log_ratio = np.log(np.maximum(prediction, EPSILON)) - log_prior[start:end]
                delta[start:end] = prediction + PRIOR_PENALTY * prediction * (
                    log_ratio - float(prediction @ log_ratio))
                delta[start + chosen[index]] -= 1
            delta *= self.adaptation_weight * MAX_RESIDUAL * (1 - squashed ** 2) / len(chosen)
            hidden_delta = np.outer(delta, self.w2) * (1 - hidden ** 2)
            gradients = [matrix.T @ hidden_delta + WEIGHT_PENALTY * self.w1,
                         hidden_delta.sum(axis=0),
                         hidden.T @ delta + WEIGHT_PENALTY * self.w2,
                         float(delta.sum())]
            norm = math.sqrt(sum(float(np.sum(gradient ** 2)) for gradient in gradients))
            step = LEARNING_RATE * min(1.0, 5.0 / max(norm, EPSILON))
            self.w1 -= step * gradients[0]
            self.b1 -= step * gradients[1]
            self.w2 -= step * gradients[2]
            self.b2 -= step * gradients[3]

    def dump(self):
        """Return independent JSON-compatible state, including choice replay."""
        return {
            "version": SCHEMA_VERSION, "feature_version": FEATURE_VERSION,
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": float(self.b2), "samples": self.samples,
            "replay": [{"fen": row["fen"], "prior": dict(row["prior"]), "chosen": row["chosen"]}
                       for row in self.replay],
        }
