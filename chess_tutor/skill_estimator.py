"""
Player Skill Estimator

Tracks player performance across games and estimates their Elo rating.
Used to dynamically adjust Maia's difficulty level.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

logger = logging.getLogger(__name__)

# Maia available levels
MAIA_LEVELS = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]


@dataclass
class MoveRecord:
    """Record of a single move's quality."""
    quality: str  # "Blunder", "Mistake", "Normal", "Good", "Excellent"
    eval_change: int  # Centipawns lost/gained


@dataclass
class GameRecord:
    """Record of a completed game."""
    result: str  # "win", "loss", "draw"
    opponent_level: int  # Maia level played against
    moves: List[MoveRecord] = field(default_factory=list)

    @property
    def blunder_count(self) -> int:
        return sum(1 for m in self.moves if m.quality == "Blunder")

    @property
    def mistake_count(self) -> int:
        return sum(1 for m in self.moves if m.quality == "Mistake")

    @property
    def good_move_count(self) -> int:
        return sum(1 for m in self.moves if m.quality in ("Good", "Excellent"))


class SkillEstimator:
    """
    Estimates player skill based on move quality and game results.

    The estimation considers:
    - Move quality distribution (blunders, mistakes, good moves)
    - Average centipawn loss per move
    - Game results against different Maia levels
    """

    # Starting Elo for new players
    DEFAULT_ELO = 1200

    # Elo adjustment bounds
    MIN_ELO = 800
    MAX_ELO = 2000

    # How many recent moves to consider for current game adjustments
    RECENT_MOVES_WINDOW = 20

    # Elo adjustments based on move quality
    MOVE_QUALITY_ADJUSTMENTS = {
        "Blunder": -15,
        "Mistake": -8,
        "Normal": 0,
        "Good": +5,
        "Excellent": +12,
    }

    # Elo adjustments based on game results (modified by opponent strength)
    RESULT_ADJUSTMENTS = {
        "win": +25,
        "draw": +5,
        "loss": -15,
    }

    def __init__(self, initial_elo: int = DEFAULT_ELO):
        self.estimated_elo = initial_elo
        self.game_history: List[GameRecord] = []
        self.current_game_moves: List[MoveRecord] = []
        self.recent_moves: deque = deque(maxlen=self.RECENT_MOVES_WINDOW)

        # Running statistics
        self.total_moves = 0
        self.total_blunders = 0
        self.total_mistakes = 0
        self.total_good_moves = 0
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def record_move(self, quality: str, eval_change: int):
        """
        Record a move and update the running Elo estimate.

        Args:
            quality: Move quality classification
            eval_change: Centipawn change from the move
        """
        move = MoveRecord(quality=quality, eval_change=eval_change)
        self.current_game_moves.append(move)
        self.recent_moves.append(move)

        # Update statistics
        self.total_moves += 1
        if quality == "Blunder":
            self.total_blunders += 1
        elif quality == "Mistake":
            self.total_mistakes += 1
        elif quality in ("Good", "Excellent"):
            self.total_good_moves += 1

        # Adjust Elo based on move quality (smaller adjustments during game)
        adjustment = self.MOVE_QUALITY_ADJUSTMENTS.get(quality, 0) * 0.3
        self._adjust_elo(adjustment)

        logger.debug(f"Move recorded: {quality}, Elo now: {self.estimated_elo}")

    def record_game_end(self, result: str, opponent_level: int):
        """
        Record the end of a game and make larger Elo adjustments.

        Args:
            result: "win", "loss", or "draw"
            opponent_level: The Maia level that was played against
        """
        game = GameRecord(
            result=result,
            opponent_level=opponent_level,
            moves=self.current_game_moves.copy()
        )
        self.game_history.append(game)
        self.current_game_moves.clear()

        # Update game statistics
        self.games_played += 1
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        else:
            self.draws += 1

        # Calculate Elo adjustment based on result and opponent strength
        base_adjustment = self.RESULT_ADJUSTMENTS.get(result, 0)

        # Modify based on opponent strength relative to player
        strength_diff = opponent_level - self.estimated_elo

        if result == "win":
            # Winning against stronger opponents gives more points
            modifier = 1.0 + (strength_diff / 400)
        elif result == "loss":
            # Losing to weaker opponents costs more
            modifier = 1.0 - (strength_diff / 400)
        else:
            modifier = 1.0

        adjustment = base_adjustment * max(0.5, min(2.0, modifier))
        self._adjust_elo(adjustment)

        # Also factor in move quality from the game
        if game.blunder_count >= 3:
            self._adjust_elo(-10)
        elif game.blunder_count == 0 and game.mistake_count <= 2:
            self._adjust_elo(+10)

        logger.info(f"Game ended: {result} vs Maia-{opponent_level}, Elo now: {self.estimated_elo}")

    def _adjust_elo(self, adjustment: float):
        """Apply an Elo adjustment with bounds checking."""
        self.estimated_elo = int(max(
            self.MIN_ELO,
            min(self.MAX_ELO, self.estimated_elo + adjustment)
        ))

    def get_recommended_maia_level(self) -> int:
        """
        Get the recommended Maia level based on current estimated Elo.
        Rounds UP to give the player a slight challenge.

        Returns:
            The Maia level to use (1100-1900)
        """
        for level in MAIA_LEVELS:
            if level >= self.estimated_elo:
                return level
        return MAIA_LEVELS[-1]  # Max level if player is very strong

    def get_stats(self) -> dict:
        """Get player statistics summary."""
        avg_cpl = 0
        if self.recent_moves:
            # Average centipawn loss (only count losses)
            losses = [abs(m.eval_change) for m in self.recent_moves if m.eval_change < 0]
            avg_cpl = sum(losses) / len(losses) if losses else 0

        win_rate = (self.wins / self.games_played * 100) if self.games_played > 0 else 0

        return {
            "estimated_elo": self.estimated_elo,
            "recommended_maia_level": self.get_recommended_maia_level(),
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": round(win_rate, 1),
            "total_moves": self.total_moves,
            "blunder_rate": round(self.total_blunders / max(1, self.total_moves) * 100, 1),
            "avg_centipawn_loss": round(avg_cpl, 1),
        }

    def get_performance_summary(self) -> str:
        """Get a human-readable performance summary."""
        stats = self.get_stats()

        lines = [
            f"Estimated Rating: {stats['estimated_elo']}",
            f"Games: {stats['games_played']} ({stats['wins']}W/{stats['losses']}L/{stats['draws']}D)",
        ]

        if stats['games_played'] > 0:
            lines.append(f"Win Rate: {stats['win_rate']}%")

        if stats['total_moves'] > 10:
            lines.append(f"Blunder Rate: {stats['blunder_rate']}%")

        return " | ".join(lines)

    def reset(self):
        """Reset all statistics (for testing or new player)."""
        self.__init__(self.DEFAULT_ELO)


# Global instance
skill_estimator = SkillEstimator()
