"""
Chess Position Analyzer

Provides rich analysis of chess positions to give the LLM meaningful context
for tutoring responses. Analyzes threats, tactics, material, game phase, and more.
"""

import chess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# Piece values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king"
}


@dataclass
class ThreatInfo:
    """Information about a threat on the board"""
    attacker_square: chess.Square
    target_square: chess.Square
    attacker_piece: chess.PieceType
    target_piece: chess.PieceType
    is_hanging: bool  # Target has no defenders


@dataclass
class TacticalPattern:
    """A detected tactical pattern"""
    pattern_type: str  # "fork", "pin", "skewer", "discovered_attack", "hanging_piece"
    description: str
    squares_involved: List[chess.Square]


@dataclass
class PositionAnalysis:
    """Complete analysis of a chess position"""
    # Material
    material_balance: int  # Positive = white ahead
    material_description: str

    # Threats
    white_threats: List[ThreatInfo]
    black_threats: List[ThreatInfo]
    threats_summary: str

    # Tactics
    tactical_patterns: List[TacticalPattern]
    tactics_summary: str

    # Position quality
    game_phase: str  # "opening", "middlegame", "endgame"
    white_king_safety: str
    black_king_safety: str
    center_control: str

    # Evaluation context
    position_assessment: str  # Human-readable position summary
    who_is_winning: str

    # Move quality (if last move provided)
    last_move_assessment: Optional[str] = None


class ChessAnalyzer:
    """Analyzes chess positions to provide rich context for the LLM tutor"""

    def __init__(self):
        # Center squares for control assessment
        self.center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        self.extended_center = [
            chess.C3, chess.C4, chess.C5, chess.C6,
            chess.D3, chess.D4, chess.D5, chess.D6,
            chess.E3, chess.E4, chess.E5, chess.E6,
            chess.F3, chess.F4, chess.F5, chess.F6
        ]

    def analyze_position(self, board: chess.Board,
                         engine_eval: Optional[int] = None,
                         last_move_eval_change: Optional[int] = None) -> PositionAnalysis:
        """
        Perform comprehensive analysis of the position.

        Args:
            board: Current chess position
            engine_eval: Engine evaluation in centipawns (positive = white better)
            last_move_eval_change: How much the eval changed after last move
        """
        # Material analysis
        material_balance = self._calculate_material(board)
        material_desc = self._describe_material(material_balance, board)

        # Threat analysis
        white_threats = self._find_threats(board, chess.WHITE)
        black_threats = self._find_threats(board, chess.BLACK)
        threats_summary = self._summarize_threats(white_threats, black_threats, board)

        # Tactical patterns
        tactics = self._find_tactics(board)
        tactics_summary = self._summarize_tactics(tactics, board)

        # Position characteristics
        game_phase = self._determine_game_phase(board)
        white_king_safety = self._assess_king_safety(board, chess.WHITE)
        black_king_safety = self._assess_king_safety(board, chess.BLACK)
        center_control = self._assess_center_control(board)

        # Overall assessment
        position_assessment = self._create_position_assessment(
            board, material_balance, engine_eval,
            white_threats, black_threats, tactics, game_phase
        )

        who_is_winning = self._determine_who_is_winning(material_balance, engine_eval)

        # Last move assessment
        last_move_assessment = None
        if last_move_eval_change is not None:
            last_move_assessment = self._assess_last_move(last_move_eval_change, board)

        return PositionAnalysis(
            material_balance=material_balance,
            material_description=material_desc,
            white_threats=white_threats,
            black_threats=black_threats,
            threats_summary=threats_summary,
            tactical_patterns=tactics,
            tactics_summary=tactics_summary,
            game_phase=game_phase,
            white_king_safety=white_king_safety,
            black_king_safety=black_king_safety,
            center_control=center_control,
            position_assessment=position_assessment,
            who_is_winning=who_is_winning,
            last_move_assessment=last_move_assessment
        )

    def _calculate_material(self, board: chess.Board) -> int:
        """Calculate material balance in centipawns (positive = white ahead)"""
        balance = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = PIECE_VALUES.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    balance += value
                else:
                    balance -= value
        return balance

    def _describe_material(self, balance: int, board: chess.Board) -> str:
        """Create human-readable material description"""
        # Count pieces
        white_pieces = {"Q": 0, "R": 0, "B": 0, "N": 0, "P": 0}
        black_pieces = {"Q": 0, "R": 0, "B": 0, "N": 0, "P": 0}

        piece_map = {chess.QUEEN: "Q", chess.ROOK: "R", chess.BISHOP: "B",
                     chess.KNIGHT: "N", chess.PAWN: "P"}

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                symbol = piece_map.get(piece.piece_type)
                if symbol:
                    if piece.color == chess.WHITE:
                        white_pieces[symbol] += 1
                    else:
                        black_pieces[symbol] += 1

        # Build description
        if abs(balance) < 50:
            status = "Material is roughly equal"
        elif balance > 0:
            pawns = balance / 100
            status = f"White is up {pawns:.1f} pawns worth of material"
        else:
            pawns = -balance / 100
            status = f"Black is up {pawns:.1f} pawns worth of material"

        # Note specific imbalances
        imbalances = []
        for piece in ["Q", "R", "B", "N"]:
            diff = white_pieces[piece] - black_pieces[piece]
            if diff > 0:
                name = {"Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}[piece]
                imbalances.append(f"White has {diff} extra {name}{'s' if diff > 1 else ''}")
            elif diff < 0:
                name = {"Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}[piece]
                imbalances.append(f"Black has {-diff} extra {name}{'s' if -diff > 1 else ''}")

        if imbalances:
            status += f" ({', '.join(imbalances)})"

        return status

    def _find_threats(self, board: chess.Board, color: chess.Color) -> List[ThreatInfo]:
        """Find all pieces that color is threatening"""
        threats = []
        opponent = not color

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                # Check what this piece attacks
                attacks = board.attacks(square)
                for target_sq in attacks:
                    target_piece = board.piece_at(target_sq)
                    if target_piece and target_piece.color == opponent:
                        # Check if target is defended
                        is_hanging = not board.is_attacked_by(opponent, target_sq)

                        threats.append(ThreatInfo(
                            attacker_square=square,
                            target_square=target_sq,
                            attacker_piece=piece.piece_type,
                            target_piece=target_piece.piece_type,
                            is_hanging=is_hanging
                        ))

        return threats

    def _summarize_threats(self, white_threats: List[ThreatInfo],
                          black_threats: List[ThreatInfo],
                          board: chess.Board) -> str:
        """Create human-readable threat summary"""
        summaries = []

        # Find the most important threats (hanging pieces, high-value targets)
        for threats, color_name, opponent_name in [
            (white_threats, "White", "Black"),
            (black_threats, "Black", "White")
        ]:
            hanging = [t for t in threats if t.is_hanging]
            high_value = [t for t in threats if t.target_piece in [chess.QUEEN, chess.ROOK]]

            if hanging:
                # Find most valuable hanging piece
                best_hanging = max(hanging, key=lambda t: PIECE_VALUES[t.target_piece])
                target_name = PIECE_NAMES[best_hanging.target_piece]
                target_sq = chess.square_name(best_hanging.target_square)
                summaries.append(f"{opponent_name}'s {target_name} on {target_sq} is hanging (undefended)!")

            elif high_value:
                best_target = max(high_value, key=lambda t: PIECE_VALUES[t.target_piece])
                target_name = PIECE_NAMES[best_target.target_piece]
                target_sq = chess.square_name(best_target.target_square)
                attacker_name = PIECE_NAMES[best_target.attacker_piece]
                summaries.append(f"{color_name}'s {attacker_name} is attacking {opponent_name}'s {target_name} on {target_sq}")

        # Check for check
        if board.is_check():
            in_check = "White" if board.turn == chess.WHITE else "Black"
            summaries.insert(0, f"{in_check} is in CHECK!")

        return " ".join(summaries) if summaries else "No immediate tactical threats."

    def _find_tactics(self, board: chess.Board) -> List[TacticalPattern]:
        """Detect tactical patterns like forks, pins, skewers"""
        patterns = []

        # Find forks (one piece attacking multiple valuable pieces)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                attacks = board.attacks(square)
                valuable_targets = []
                for target_sq in attacks:
                    target = board.piece_at(target_sq)
                    if target and target.color != piece.color:
                        if target.piece_type in [chess.KING, chess.QUEEN, chess.ROOK]:
                            valuable_targets.append((target_sq, target.piece_type))

                if len(valuable_targets) >= 2:
                    piece_name = PIECE_NAMES[piece.piece_type]
                    targets_desc = " and ".join([
                        f"{PIECE_NAMES[t[1]]} on {chess.square_name(t[0])}"
                        for t in valuable_targets
                    ])
                    color = "White" if piece.color == chess.WHITE else "Black"
                    patterns.append(TacticalPattern(
                        pattern_type="fork",
                        description=f"{color}'s {piece_name} on {chess.square_name(square)} is forking the {targets_desc}!",
                        squares_involved=[square] + [t[0] for t in valuable_targets]
                    ))

        # Find pins (piece that can't move because it would expose a more valuable piece)
        patterns.extend(self._find_pins(board))

        # Find hanging pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.PAWN:
                opponent = not piece.color
                if board.is_attacked_by(opponent, square) and not board.is_attacked_by(piece.color, square):
                    piece_name = PIECE_NAMES[piece.piece_type]
                    color = "White" if piece.color == chess.WHITE else "Black"
                    patterns.append(TacticalPattern(
                        pattern_type="hanging_piece",
                        description=f"{color}'s {piece_name} on {chess.square_name(square)} is hanging (attacked but not defended)!",
                        squares_involved=[square]
                    ))

        return patterns

    def _find_pins(self, board: chess.Board) -> List[TacticalPattern]:
        """Find pinned pieces"""
        patterns = []

        for color in [chess.WHITE, chess.BLACK]:
            king_sq = board.king(color)
            if king_sq is None:
                continue

            # Check all opponent sliding pieces
            opponent = not color
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == opponent:
                    if piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                        # Check if there's exactly one piece between attacker and king
                        ray = chess.SquareSet.ray(square, king_sq)
                        if ray:
                            pieces_between = []
                            for ray_sq in ray:
                                if ray_sq != square and ray_sq != king_sq:
                                    p = board.piece_at(ray_sq)
                                    if p:
                                        pieces_between.append((ray_sq, p))

                            # If exactly one piece of the target color is between
                            if len(pieces_between) == 1:
                                pinned_sq, pinned_piece = pieces_between[0]
                                if pinned_piece.color == color:
                                    attacker_name = PIECE_NAMES[piece.piece_type]
                                    pinned_name = PIECE_NAMES[pinned_piece.piece_type]
                                    attacker_color = "White" if opponent == chess.WHITE else "Black"
                                    pinned_color = "White" if color == chess.WHITE else "Black"
                                    patterns.append(TacticalPattern(
                                        pattern_type="pin",
                                        description=f"{pinned_color}'s {pinned_name} on {chess.square_name(pinned_sq)} is pinned to the king by {attacker_color}'s {attacker_name}!",
                                        squares_involved=[square, pinned_sq, king_sq]
                                    ))

        return patterns

    def _summarize_tactics(self, tactics: List[TacticalPattern], board: chess.Board) -> str:
        """Create tactical summary"""
        if not tactics:
            return "No major tactical patterns detected."

        # Prioritize: forks > pins > hanging pieces
        priority = {"fork": 1, "pin": 2, "skewer": 3, "hanging_piece": 4}
        sorted_tactics = sorted(tactics, key=lambda t: priority.get(t.pattern_type, 99))

        # Return top 2-3 most important
        return " ".join([t.description for t in sorted_tactics[:3]])

    def _determine_game_phase(self, board: chess.Board) -> str:
        """Determine if we're in opening, middlegame, or endgame"""
        # Count total material (excluding kings)
        total_material = 0
        queens = 0
        minor_pieces = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                total_material += PIECE_VALUES.get(piece.piece_type, 0)
                if piece.piece_type == chess.QUEEN:
                    queens += 1
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    minor_pieces += 1

        # Check move count for opening
        move_count = len(board.move_stack)

        if move_count < 10 and total_material > 5000:
            return "opening"
        elif total_material < 2600 or (queens == 0 and minor_pieces <= 2):
            return "endgame"
        else:
            return "middlegame"

    def _assess_king_safety(self, board: chess.Board, color: chess.Color) -> str:
        """Assess king safety for a color"""
        king_sq = board.king(color)
        if king_sq is None:
            return "unknown"

        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)

        # Check if castled
        is_castled = False
        for move in board.move_stack:
            if board.is_castling(move):
                # This is approximate - check if this color castled
                from_sq = move.from_square
                piece_moved = chess.KING
                if chess.square_rank(from_sq) == (0 if color == chess.WHITE else 7):
                    is_castled = True
                    break

        # Count attackers near king
        opponent = not color
        attacks_near_king = 0
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                check_rank = king_rank + dr
                check_file = king_file + df
                if 0 <= check_rank <= 7 and 0 <= check_file <= 7:
                    sq = chess.square(check_file, check_rank)
                    if board.is_attacked_by(opponent, sq):
                        attacks_near_king += 1

        # Count pawn shield
        pawn_shield = 0
        shield_rank = king_rank + (1 if color == chess.WHITE else -1)
        if 0 <= shield_rank <= 7:
            for df in [-1, 0, 1]:
                check_file = king_file + df
                if 0 <= check_file <= 7:
                    sq = chess.square(check_file, shield_rank)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        pawn_shield += 1

        color_name = "White" if color == chess.WHITE else "Black"

        if board.is_check():
            return f"{color_name}'s king is in CHECK!"
        elif is_castled and pawn_shield >= 2 and attacks_near_king < 3:
            return f"{color_name}'s king is safe (castled with good pawn cover)"
        elif king_rank in [0, 7] and pawn_shield >= 2:
            return f"{color_name}'s king is reasonably safe"
        elif attacks_near_king >= 5:
            return f"{color_name}'s king is EXPOSED and under heavy attack!"
        elif pawn_shield == 0 and king_rank not in [0, 7]:
            return f"{color_name}'s king is exposed in the center - dangerous!"
        else:
            return f"{color_name}'s king safety is moderate"

    def _assess_center_control(self, board: chess.Board) -> str:
        """Assess who controls the center"""
        white_control = 0
        black_control = 0

        for sq in self.center_squares:
            piece = board.piece_at(sq)
            if piece:
                if piece.color == chess.WHITE:
                    white_control += 2  # Occupation is strong
                else:
                    black_control += 2

            # Also count attacks on center
            if board.is_attacked_by(chess.WHITE, sq):
                white_control += 1
            if board.is_attacked_by(chess.BLACK, sq):
                black_control += 1

        diff = white_control - black_control

        if abs(diff) <= 2:
            return "Center control is roughly equal"
        elif diff > 4:
            return "White has strong control of the center"
        elif diff > 0:
            return "White has a slight edge in center control"
        elif diff < -4:
            return "Black has strong control of the center"
        else:
            return "Black has a slight edge in center control"

    def _determine_who_is_winning(self, material: int, engine_eval: Optional[int]) -> str:
        """Determine who is winning based on eval and material"""
        # Prefer engine eval if available
        eval_to_use = engine_eval if engine_eval is not None else material

        if abs(eval_to_use) < 50:
            return "The position is roughly equal"
        elif eval_to_use > 300:
            return "White is winning"
        elif eval_to_use > 150:
            return "White has a significant advantage"
        elif eval_to_use > 50:
            return "White is slightly better"
        elif eval_to_use < -300:
            return "Black is winning"
        elif eval_to_use < -150:
            return "Black has a significant advantage"
        else:
            return "Black is slightly better"

    def _assess_last_move(self, eval_change: int, board: chess.Board) -> str:
        """Assess the quality of the last move based on eval change"""
        # Note: eval_change is from the perspective of the player who moved
        # Positive = good for them, negative = bad

        who_moved = "Black" if board.turn == chess.WHITE else "White"

        if eval_change <= -300:
            return f"BLUNDER! {who_moved}'s last move was a serious mistake that may have lost the game!"
        elif eval_change <= -150:
            return f"MISTAKE! {who_moved}'s last move significantly worsened their position."
        elif eval_change <= -75:
            return f"Inaccuracy. {who_moved}'s last move was not optimal."
        elif eval_change >= 200:
            return f"Brilliant! {who_moved}'s last move was exceptional and dramatically improved their position!"
        elif eval_change >= 100:
            return f"Great move! {who_moved}'s last move significantly improved their position."
        elif eval_change >= 50:
            return f"Good move by {who_moved}."
        else:
            return f"{who_moved}'s last move was solid."

    def _create_position_assessment(self, board: chess.Board, material: int,
                                    engine_eval: Optional[int],
                                    white_threats: List[ThreatInfo],
                                    black_threats: List[ThreatInfo],
                                    tactics: List[TacticalPattern],
                                    game_phase: str) -> str:
        """Create an overall position assessment"""
        parts = []

        # Game phase context
        phase_context = {
            "opening": "We're still in the opening. Focus on development and controlling the center.",
            "middlegame": "We're in the middlegame. Time to create threats and find tactics!",
            "endgame": "We've reached the endgame. King activity and passed pawns are crucial."
        }
        parts.append(phase_context.get(game_phase, ""))

        # Winning status
        who_winning = self._determine_who_is_winning(material, engine_eval)
        parts.append(who_winning + ".")

        # Most critical threat or tactic
        critical_tactics = [t for t in tactics if t.pattern_type in ["fork", "pin", "hanging_piece"]]
        if critical_tactics:
            parts.append(f"IMPORTANT: {critical_tactics[0].description}")

        # Hanging pieces warning
        white_hanging = [t for t in black_threats if t.is_hanging]
        black_hanging = [t for t in white_threats if t.is_hanging]

        if white_hanging:
            parts.append("Warning: White has hanging piece(s)!")
        if black_hanging:
            parts.append("Warning: Black has hanging piece(s)!")

        return " ".join(parts)

    def get_analysis_for_llm(self, board: chess.Board,
                            engine_eval: Optional[int] = None,
                            last_move_eval_change: Optional[int] = None) -> str:
        """
        Get a formatted analysis string ready to be included in LLM prompts.
        This is the main method to call for getting context for the tutor.
        """
        analysis = self.analyze_position(board, engine_eval, last_move_eval_change)

        sections = []

        # Position overview
        sections.append(f"=== POSITION ANALYSIS ===")
        sections.append(f"Game Phase: {analysis.game_phase.upper()}")
        sections.append(f"Position Assessment: {analysis.who_is_winning}")
        sections.append(f"Material: {analysis.material_description}")

        # Last move quality (if available)
        if analysis.last_move_assessment:
            sections.append(f"\n=== LAST MOVE ===")
            sections.append(analysis.last_move_assessment)

        # Threats and tactics
        if analysis.threats_summary != "No immediate tactical threats.":
            sections.append(f"\n=== THREATS ===")
            sections.append(analysis.threats_summary)

        if analysis.tactics_summary != "No major tactical patterns detected.":
            sections.append(f"\n=== TACTICS ===")
            sections.append(analysis.tactics_summary)

        # King safety (only if noteworthy)
        sections.append(f"\n=== KING SAFETY ===")
        sections.append(f"White: {analysis.white_king_safety}")
        sections.append(f"Black: {analysis.black_king_safety}")

        # Center control
        sections.append(f"\n=== POSITIONAL ===")
        sections.append(analysis.center_control)

        return "\n".join(sections)


# Global instance for convenience
chess_analyzer = ChessAnalyzer()
