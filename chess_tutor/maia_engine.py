"""
Maia Chess Engine wrapper.

Maia is a human-like chess engine built on Leela Chess Zero (LC0).
It plays at various human skill levels (1100-1900 Elo).

Setup: Run `python scripts/setup_maia.py` to download weights.
"""

import chess
import chess.engine
import logging
import os
import shutil
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Available Maia skill levels (Elo ratings)
MAIA_LEVELS = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
DEFAULT_LEVEL = 1100


class MaiaEngine:
    """
    Wrapper for the Maia chess engine.

    Maia uses LC0 (Leela Chess Zero) with specially trained neural network
    weights that mimic human play at various skill levels.
    """

    # Evaluation thresholds (in centipawns)
    BLUNDER_THRESHOLD = -200   # -2 pawns
    MISTAKE_THRESHOLD = -100   # -1 pawn
    GOOD_MOVE_THRESHOLD = 50   # +0.5 pawns
    EXCELLENT_MOVE_THRESHOLD = 150  # +1.5 pawns

    def __init__(self, project_dir: str, level: int = DEFAULT_LEVEL):
        """
        Initialize the Maia engine.

        Args:
            project_dir: Path to the project root directory
            level: Maia skill level (1100-1900 Elo)
        """
        self.project_dir = project_dir
        self.level = self._validate_level(level)
        self.weights_dir = os.path.join(project_dir, "maia-chess", "maia_weights")
        self.engine: Optional[chess.engine.SimpleEngine] = None

        self._load_engine()

    def _validate_level(self, level: int) -> int:
        """Validate and return the closest valid Maia level."""
        if level not in MAIA_LEVELS:
            closest = min(MAIA_LEVELS, key=lambda x: abs(x - level))
            logger.warning(f"Invalid Maia level {level}, using {closest}")
            return closest
        return level

    def _get_lc0_path(self) -> str:
        """Find the LC0 executable."""
        lc0_path = shutil.which("lc0")
        if lc0_path:
            return lc0_path

        # Check common locations
        common_paths = [
            "/usr/local/bin/lc0",
            "/usr/bin/lc0",
            os.path.expanduser("~/.local/bin/lc0"),
        ]
        for path in common_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        raise RuntimeError(
            "LC0 not found. Please install it:\n"
            "  macOS:   brew install lc0\n"
            "  Ubuntu:  sudo apt install lc0\n"
            "  Or run:  python scripts/setup_maia.py"
        )

    def _load_engine(self):
        """Initialize the chess engine with Maia weights."""
        weights_file = f"maia-{self.level}.pb.gz"
        weights_path = os.path.join(self.weights_dir, weights_file)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Maia weights not found at {weights_path}\n"
                f"Run: python scripts/setup_maia.py --weight maia-{self.level}"
            )

        engine_path = self._get_lc0_path()
        logger.info(f"Loading Maia-{self.level} engine from {engine_path}")

        self.engine = chess.engine.SimpleEngine.popen_uci([
            engine_path,
            f"--weights={weights_path}"
        ])

    def get_best_move(self, board: chess.Board, time_limit: float = 1.0) -> chess.Move:
        """
        Get the best move for the given position.

        Args:
            board: Current chess position
            time_limit: Time in seconds for engine to think

        Returns:
            The best move according to Maia
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized")

        result = self.engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move

    def get_position_evaluation(self, board: chess.Board, time_limit: float = 1.0) -> int:
        """
        Get numerical evaluation of the position.

        Args:
            board: Position to evaluate
            time_limit: Analysis time in seconds

        Returns:
            Evaluation in centipawns (positive = white is better)
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized")

        info = self.engine.analyse(board, chess.engine.Limit(time=time_limit))

        # Handle both single result and list format
        if isinstance(info, list):
            info = info[0]

        score = info["score"].white()
        return score.score(mate_score=10000)

    def get_top_moves(self, board: chess.Board, num_moves: int = 5,
                      time_limit: float = 1.0) -> List[Dict]:
        """
        Get the top N moves with evaluations.

        Args:
            board: Position to analyze
            num_moves: Number of moves to return
            time_limit: Analysis time in seconds

        Returns:
            List of dicts with move info: {move, san, evaluation, mate}
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized")

        analysis = self.engine.analyse(
            board,
            chess.engine.Limit(time=time_limit),
            multipv=num_moves
        )

        moves = []
        for pv in analysis:
            if "pv" in pv and pv["pv"]:
                score = pv["score"].white()
                moves.append({
                    "move": pv["pv"][0],
                    "san": board.san(pv["pv"][0]),
                    "evaluation": score.score(mate_score=10000),
                    "mate": score.mate()
                })

        return moves

    def evaluate_move_quality(self, board: chess.Board, move: chess.Move,
                              time_limit: float = 1.0) -> Dict:
        """
        Evaluate how good a move is compared to the best move.

        Args:
            board: Position before the move
            move: The move to evaluate
            time_limit: Analysis time in seconds

        Returns:
            Dict with quality assessment and evaluation change
        """
        # Evaluation before the move
        initial_eval = self.get_position_evaluation(board, time_limit)

        # Make move on a copy
        board_copy = board.copy()
        board_copy.push(move)

        # Evaluation after (negated since it's opponent's turn)
        new_eval = -self.get_position_evaluation(board_copy, time_limit)

        # Calculate change from moving player's perspective
        eval_diff = new_eval - initial_eval
        quality = self._classify_move_quality(eval_diff)

        return {
            "quality": quality,
            "evaluation_difference": eval_diff,
            "absolute_evaluation": new_eval
        }

    def _classify_move_quality(self, eval_diff: int) -> str:
        """Classify move quality based on evaluation change."""
        if eval_diff <= self.BLUNDER_THRESHOLD:
            return "Blunder"
        elif eval_diff <= self.MISTAKE_THRESHOLD:
            return "Mistake"
        elif eval_diff >= self.EXCELLENT_MOVE_THRESHOLD:
            return "Excellent"
        elif eval_diff >= self.GOOD_MOVE_THRESHOLD:
            return "Good"
        return "Normal"

    def set_level(self, level: int):
        """
        Change the Maia skill level.

        Args:
            level: New skill level (1100-1900)
        """
        new_level = self._validate_level(level)
        if new_level != self.level:
            self.close()
            self.level = new_level
            self._load_engine()

    def close(self):
        """Close the engine process and free resources."""
        if self.engine:
            try:
                self.engine.quit()
            except Exception as e:
                logger.warning(f"Error closing engine: {e}")
            finally:
                self.engine = None

    def __del__(self):
        self.close()

    def __repr__(self):
        return f"MaiaEngine(level={self.level})"
