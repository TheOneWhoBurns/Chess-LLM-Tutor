"""Pretrained human move probabilities from Maia's policy head through LC0.

LC0 prints the complete root policy in VerboseMoveStats. Consume the stream,
not ``analyse()``'s final dictionary (which retains only the last info string).
There is deliberately no heuristic or uniform fallback when the model fails.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shutil
import threading

import chess
import chess.engine


DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / ".runtime/maia/maia-1500.pb.gz"
# A printed 0.00% represents [0, 0.005%) before percent rounding. Use its
# midpoint as a quantization-aware estimate, rather than treating it as zero.
POLICY_FLOOR = 0.000025
POLICY_QUANTIZATION_VERSION = "lc0-percent-midpoint-v2"
_MOVE_STATS = re.compile(
    r"^\s*([a-h][1-8][a-h][1-8][qrbn]?)\s+\([^)]*\).*?"
    r"\(P:\s*([0-9]+(?:\.[0-9]+)?)%\)"
)


class MaiaUnavailable(RuntimeError):
    """The real pretrained policy could not be loaded or evaluated."""


def parse_policy_lines(board: chess.Board, lines: list[str]) -> dict[str, float]:
    """Validate a complete LC0 root policy and normalize its rounded output.

    Verbose stats always encode castling as king-to-rook, irrespective of
    UCI_Chess960. python-chess converts these to the board's normal UCI form.
    A printed 0.00% is numerical rounding, so keep that legal move trainable.
    Missing moves, by contrast, indicate a failed/incomplete inference.
    """
    legal = {move.uci() for move in board.legal_moves}
    if not legal:
        return {}
    observed: dict[str, float] = {}
    for line in lines:
        match = _MOVE_STATS.match(line)
        if not match:
            continue
        raw_move, percent = match.groups()
        try:
            move = board.parse_uci(raw_move).uci()
        except ValueError as exc:
            raise MaiaUnavailable(f"LC0 returned an illegal policy move: {raw_move}") from exc
        probability = float(percent) / 100
        if move not in legal or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise MaiaUnavailable(f"LC0 returned an invalid policy value for {raw_move}.")
        observed[move] = probability
    missing = legal - observed.keys()
    if missing:
        raise MaiaUnavailable(
            f"LC0 returned an incomplete Maia policy: {len(observed)}/{len(legal)} legal moves."
        )
    mass = math.fsum(observed.values())
    # LC0 rounds each probability to 0.01 percentage points. Allow slightly
    # more than the maximum resulting error, but reject truncated/corrupt mass.
    if abs(mass - 1) > len(legal) * 0.000051 + 0.001:
        raise MaiaUnavailable(f"LC0 policy probabilities have invalid total mass: {mass:.6f}.")
    floored = {move: observed[move] if observed[move] > 0 else POLICY_FLOOR
               for move in sorted(legal)}
    total = math.fsum(floored.values())
    return {move: value / total for move, value in floored.items()}


class MaiaPolicy:
    """Lazily load a single local Maia network; callers serialize access.

    The frozen network predicts human choices. Personalization belongs to the
    player policy adapter that consumes these priors, not this runtime wrapper.
    """

    def __init__(self, weights_path=None, engine_path=None):
        self.weights_path = Path(
            weights_path or os.environ.get("NEMESIS_MAIA_WEIGHTS") or DEFAULT_WEIGHTS
        ).expanduser().resolve()
        self.engine_path = str(
            engine_path
            or os.environ.get("NEMESIS_LC0_ENGINE")
            or shutil.which("lc0")
            or "lc0"
        )
        self.backend = os.environ.get("NEMESIS_MAIA_BACKEND", "blas")
        self.timeout = 15.0
        self._engine: chess.engine.SimpleEngine | None = None
        self._engine_version: str | None = None
        self._last_error: str | None = None

    @property
    def info(self) -> dict:
        return {
            "name": "Maia 1500" if self.weights_path.name == "maia-1500.pb.gz" else "Maia neural policy",
            "weights": str(self.weights_path),
            "version": self._engine_version or "Maia v1.0; LC0 not loaded",
            "backend": self.backend,
            "quantization": POLICY_QUANTIZATION_VERSION,
            "ready": self._engine is not None,
            "weights_present": self.weights_path.is_file(),
            "error": self._last_error,
        }

    @property
    def status(self) -> dict:
        return self.info

    def _start(self) -> chess.engine.SimpleEngine:
        if self._engine is not None:
            return self._engine
        if not self.weights_path.is_file():
            raise MaiaUnavailable(
                f"Maia weights are missing: {self.weights_path}. See docs/MAIA_RUNTIME.md."
            )
        command = [
            self.engine_path, "classic", "--config=" + os.devnull,
            f"--weights={self.weights_path}", f"--backend={self.backend}",
            "--verbose-move-stats", "--policy-softmax-temp=1", "--threads=1",
            "--minibatch-size=1", "--max-prefetch=0", "--nncache=10000",
            "--cache-history-length=7", "--history-fill-new=fen_only",
        ]
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(command, timeout=self.timeout)
        except (OSError, chess.engine.EngineError, TimeoutError) as exc:
            raise MaiaUnavailable(
                f"Could not start LC0 at {self.engine_path}: {exc}. See docs/MAIA_RUNTIME.md."
            ) from exc
        self._engine_version = self._engine.id.get("name", "LC0")
        return self._engine

    def probabilities(self, board: chess.Board) -> dict[str, float]:
        if not board.is_valid():
            raise ValueError("Maia requires a valid standard chess position.")
        if not any(board.legal_moves):
            return {}
        engine = None
        timer = None
        timed_out = threading.Event()
        try:
            engine = self._start()

            def expire():
                # SimpleAnalysisResult iteration does not enforce engine.timeout.
                # Closing the transport also kills a hung child and unblocks it.
                timed_out.set()
                engine.close()

            timer = threading.Timer(self.timeout, expire)
            timer.daemon = True
            timer.start()
            lines = []
            # A new game token clears the search tree. Actual board history is
            # still supplied by python-chess's position command to the network.
            with engine.analysis(
                board, chess.engine.Limit(nodes=1), game=object(), info=chess.engine.INFO_ALL
            ) as analysis:
                for item in analysis:
                    if "string" in item:
                        lines.append(item["string"])
            timer.cancel()
            timer.join()
            if timed_out.is_set():
                raise TimeoutError("Maia inference exceeded its deadline")
            result = parse_policy_lines(board, lines)
            self._last_error = None
            return result
        except (OSError, chess.engine.EngineError, TimeoutError, MaiaUnavailable) as exc:
            if timed_out.is_set():
                message = f"Maia inference timed out after {self.timeout:g} seconds."
            else:
                message = str(exc)
            self._last_error = message
            self.close()
            if isinstance(exc, MaiaUnavailable) and not timed_out.is_set():
                raise
            raise MaiaUnavailable(f"Maia policy unavailable: {message}") from exc
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()

    def close(self):
        engine, self._engine = self._engine, None
        if engine is not None:
            # close() closes the transport without waiting on an unresponsive
            # UCI 'quit'; no worker subprocess is retained on a failed request.
            engine.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
