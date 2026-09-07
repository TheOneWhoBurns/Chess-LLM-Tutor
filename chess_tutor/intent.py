# intent.py
from typing import Dict, Optional
import re
import chess
from .legacy_models import model_manager

class IntentClassifier:
    INTENTS = [
        "make_move",
        "ask_explanation",
        "request_game",
        "general_chat",
        "quit_game"
    ]

    def __init__(self):
        self._move_patterns = [
            # Add pattern for long algebraic notation (e.g. e2-e4, b1-c3)
            r'\b[a-h][1-8]-[a-h][1-8]\b',
            # Existing patterns
            r'\b[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?\+?\#?\b',
            r'\b[NBRQK][a-h1-8][a-h][1-8]\b',
            r'\b[a-h][1-8]\b',
            r'\bO-O(-O)?\b'
        ]
        self._move_regex = re.compile('|'.join(self._move_patterns))

        # Add patterns for advice requests
        self._advice_patterns = [
            r'\b(?:what|suggest|recommend|advice|help|unsure|uncertain)\b.*\b(?:move|play|next)\b',
            r'\b(?:good|best)\s+(?:move|continuation)\b',
            r'\bshould\s+(?:i|we)\s+(?:move|play)\b'
        ]
        self._advice_regex = re.compile('|'.join(self._advice_patterns), re.IGNORECASE)

        self.board = chess.Board()

    def classify(self, message: str) -> Dict[str, any]:
        """Classify intent and extract move if present"""
        # Check for move first (fastest)
        move = self.extract_move(message)
        if move and self.validate_move(move):
            return {
                "intent": "make_move",
                "move": move,
                "confidence": 0.9
            }

        # Check for advice requests before using the model
        if self._advice_regex.search(message):
            return {
                "intent": "ask_explanation",
                "move": None,
                "confidence": 0.9
            }

        # Then use model for other intents
        result = model_manager.get_intent(message, self.INTENTS)
        return {
            "intent": result["labels"][0],
            "move": None,
            "confidence": result["scores"][0]
        }

    def convert_long_algebraic(self, move_str: str) -> Optional[str]:
        """Convert long algebraic notation (e.g. 'e2-e4', 'b1-c3') to SAN"""
        if '-' not in move_str:
            return move_str

        from_square, to_square = move_str.split('-')
        try:
            from_sq = chess.parse_square(from_square.lower())
            to_sq = chess.parse_square(to_square.lower())

            # Create the move and verify it's legal
            move = chess.Move(from_sq, to_sq)
            if move in self.board.legal_moves:
                return self.board.san(move)
        except ValueError:
            pass

        return None

    def extract_potential_moves(self, text: str) -> list[str]:
        """Extract potential chess moves using regex"""
        moves = []
        for match in self._move_regex.finditer(text):
            move = match.group()
            # Convert long algebraic notation if needed
            if '-' in move:
                converted = self.convert_long_algebraic(move)
                if converted:
                    moves.append(converted)
            else:
                moves.append(move)
        return moves

    def validate_move(self, move: str) -> bool:
        """Check if a move is legal in the current position"""
        try:
            chess_move = self.board.parse_san(move)
            return chess_move in self.board.legal_moves
        except ValueError:
            return False

    def extract_move(self, message: str) -> Optional[str]:
        """Extract and validate chess move from message"""
        potential_moves = self.extract_potential_moves(message)
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]

        for move in potential_moves:
            if move.upper() in [m.upper() for m in legal_moves]:
                return move

        context = f"Legal moves are: {', '.join(legal_moves)}. Message: {message}"
        move = model_manager.extract_move(message, context)

        if move and move.upper() in [m.upper() for m in legal_moves]:
            return move

        return None

    def update_board(self, board: chess.Board):
        """Update the internal board state"""
        self.board = board.copy()

# Global instance
intent_classifier = IntentClassifier()

def categorize_intent(message: str) -> Dict[str, any]:
    """Wrapper for backwards compatibility"""
    return intent_classifier.classify(message)
