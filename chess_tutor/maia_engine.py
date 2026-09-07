import chess
import chess.engine
import os
import subprocess

class MaiaEngine:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.weights_dir = os.path.join(project_dir, "maia-chess", "maia_weights")
        self.engine_path = self._get_lc0_path()
        self.engine = None

        # Evaluation thresholds (in centipawns)
        self.BLUNDER_THRESHOLD = -200  # -2 pawns
        self.MISTAKE_THRESHOLD = -100   # -1 pawn
        self.GOOD_MOVE_THRESHOLD = 50   # +0.5 pawns
        self.EXCELLENT_MOVE_THRESHOLD = 150  # +1.5 pawns

        self._load_engine()

    def _get_lc0_path(self):
        try:
            return subprocess.check_output(["which", "lc0"]).decode().strip()
        except subprocess.CalledProcessError:
            raise RuntimeError("lc0 not found in PATH. Make sure it's installed correctly.")

    def _load_engine(self):
        weights_file = "maia-1100.pb.gz"
        weights_path = os.path.join(self.weights_dir, weights_file)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Maia weights not found at {weights_path}")

        self.engine = chess.engine.SimpleEngine.popen_uci([
            self.engine_path,
            f"--weights={weights_path}"
        ])

    def get_best_move(self, board, time_limit=1.0):
        """
        Get the best move for the given board position using Maia.

        Args:
        board (chess.Board): The current board position
        time_limit (float): Time limit for the engine to think, in seconds

        Returns:
        chess.Move: The best move according to Maia
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized")

        result = self.engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move

    def get_position_evaluation(self, board, time_limit=1.0):
        """Get numerical evaluation of current position in centipawns"""
        info = self.engine.analyse(board, chess.engine.Limit(time=time_limit), multipv=1)
        # When using multipv, analysis returns a list of dictionaries
        if isinstance(info, list):
            info = info[0]  # Get first (and only) analysis
        return info["score"].white().score(mate_score=10000)

    def get_top_moves(self, board, num_moves=5, time_limit=1.0):
        """
        Get the top N moves for the current position with their evaluations.

        Args:
        board (chess.Board): The current board position
        num_moves (int): Number of top moves to return
        time_limit (float): Time limit for analysis, in seconds

        Returns:
        List[Dict]: List of moves with their evaluations, sorted by strength
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
            moves.append({
                "move": pv["pv"][0],  # The actual move
                "san": board.san(pv["pv"][0]),  # Move in algebraic notation
                "evaluation": pv["score"].white().score(mate_score=10000),
                "mate": pv["score"].white().mate()
            })

        return moves

    def evaluate_move_quality(self, board, move, time_limit=1.0):
        """
        Evaluate the quality of a move by comparing it to the best move.

        Args:
        board (chess.Board): The current board position
        move (chess.Move): The move to evaluate
        time_limit (float): Time limit for analysis, in seconds

        Returns:
        Dict: Move quality assessment with evaluation difference
        """
        # Get position evaluation before move
        initial_eval = self.get_position_evaluation(board, time_limit)

        # Make the move on a copy of the board
        board_copy = board.copy()
        board_copy.push(move)

        # Get evaluation after move
        new_eval = self.get_position_evaluation(board_copy, time_limit)

        # Calculate evaluation difference
        eval_diff = (new_eval - initial_eval) * (1 if board.turn == chess.WHITE else -1)

        # Determine move quality
        quality = self._get_move_quality(eval_diff)

        return {
            "quality": quality,
            "evaluation_difference": eval_diff,
            "absolute_evaluation": new_eval
        }

    def _get_move_quality(self, eval_diff):
        """
        Determine move quality based on evaluation difference.

        Args:
        eval_diff (float): The difference in evaluation after the move

        Returns:
        str: Quality assessment of the move
        """
        if eval_diff <= self.BLUNDER_THRESHOLD:
            return "Blunder"
        elif eval_diff <= self.MISTAKE_THRESHOLD:
            return "Mistake"
        elif eval_diff >= self.EXCELLENT_MOVE_THRESHOLD:
            return "Excellent"
        elif eval_diff >= self.GOOD_MOVE_THRESHOLD:
            return "Good"
        else:
            return "Normal"

    def close(self):
        """
        Close the engine process.
        """
        if self.engine:
            self.engine.quit()
            self.engine = None

    def __del__(self):
        self.close()
