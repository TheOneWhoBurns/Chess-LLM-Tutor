"""
Chess Logic Unit - Core game orchestration for the chess tutor.
Manages game state, move processing, and coordinates with engine and LLM.
"""

import logging
from typing import List, Dict, Optional
import chess

from .maia_engine import MaiaEngine
from .PromptMaker import PromptMaker
from .chess_analyzer import ChessAnalyzer
from .models import model_manager

# Set up logging
logger = logging.getLogger(__name__)


class ChessLogicUnit:
    def __init__(self, project_dir: Optional[str] = None):
        self.board = chess.Board()
        self.move_history: List[str] = []
        self.chat_history: List[Dict[str, str]] = []
        self.game_in_progress = False
        self.player_color = chess.WHITE  # Player is always white

        # Initialize components
        self.prompt_maker = PromptMaker()
        self.analyzer = ChessAnalyzer()

        if project_dir:
            try:
                self.maia_engine = MaiaEngine(project_dir)
            except Exception as e:
                logger.error(f"Failed to initialize Maia engine: {e}")
                self.maia_engine = None

        # Track evaluation for move quality assessment
        self._last_eval: Optional[int] = None

    def get_move_history(self) -> List[str]:
        """Get the history of moves"""
        return self.move_history

    def get_current_position(self) -> str:
        """Get current FEN position"""
        return self.board.fen()

    def _is_player_turn(self) -> bool:
        """Check if it's the player's turn"""
        return self.board.turn == self.player_color

    def _get_position_analysis(self) -> str:
        """Get rich position analysis for LLM context"""
        try:
            engine_eval = None
            if self.maia_engine:
                engine_eval = self.maia_engine.get_position_evaluation(self.board)

            return self.analyzer.get_analysis_for_llm(
                self.board,
                engine_eval=engine_eval
            )
        except Exception as e:
            logger.error(f"Error getting position analysis: {e}")
            return ""

    def _get_move_quality(self, move: chess.Move) -> Optional[str]:
        """Assess the quality of a move by comparing evals before/after"""
        if not self.maia_engine:
            return None

        try:
            # Get eval before the move
            temp_board = self.board.copy()
            temp_board.pop()  # Undo the move to get pre-move position
            eval_before = self.maia_engine.get_position_evaluation(temp_board)

            # Get eval after (current position, but from opponent's perspective)
            eval_after = -self.maia_engine.get_position_evaluation(self.board)

            # Calculate change from moving player's perspective
            eval_change = eval_after - eval_before

            # Use analyzer to describe the move quality
            return self.analyzer._assess_last_move(eval_change, self.board)
        except Exception as e:
            logger.error(f"Error assessing move quality: {e}")
            return None

    def handle_message(self, intent_result: Dict) -> Dict:
        """Main entry point for processing messages"""
        intent = intent_result["intent"]
        message = intent_result["message"]
        move = intent_result.get("move")

        # Handle game management first
        if intent == "request_game":
            self._reset_game()
            return {
                "status": "success",
                "message": self.prompt_maker.create_game_start_response(),
                "moves": []
            }

        if not self.game_in_progress and message.lower() in ['yes', 'sure', 'okay', 'play', 'play again', 'new game']:
            self._reset_game()
            return {
                "status": "success",
                "message": self.prompt_maker.create_game_start_response(),
                "moves": []
            }

        if intent == "quit_game":
            self.game_in_progress = False
            return {
                "status": "success",
                "message": "Game ended. Thanks for playing!",
                "moves": self.move_history
            }

        # Check if game is in progress for other intents
        if not self.game_in_progress:
            return {
                "status": "error",
                "message": self.prompt_maker.create_no_game_response(),
                "moves": []
            }

        # Handle move intent with turn checking
        if intent == "make_move" and move:
            if not self._is_player_turn():
                return {
                    "status": "error",
                    "message": "It's not your turn yet.",
                    "moves": self.move_history
                }
            return self._handle_move(message, move)

        elif intent == "ask_explanation":
            return self._handle_explanation(message)
        elif intent == "general_chat":
            return self._handle_chat(message)
        else:
            return self._handle_unknown(message)

    def _make_move(self, move_str: str) -> bool:
        """Make a move on the board with improved move parsing"""
        try:
            # Check for same square move
            if len(move_str) >= 4 and move_str[:2] == move_str[2:4]:
                return False

            # Clean up the move string
            move_str = move_str.replace('-', '').strip()

            # Try parsing as SAN first (e4, Nf3 format)
            try:
                move = self.board.parse_san(move_str)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.move_history.append(move_str)
                    return True
            except ValueError:
                pass

            # Check for castling notation (O-O or O-O-O)
            if move_str.upper() in ['OO', 'OOO', 'O-O', 'O-O-O']:
                is_kingside = len(move_str.replace('-', '')) <= 2
                if is_kingside:
                    castle_move = 'e1g1' if self.board.turn else 'e8g8'
                else:
                    castle_move = 'e1c1' if self.board.turn else 'e8c8'
                try:
                    move = chess.Move.from_uci(castle_move)
                    if move in self.board.legal_moves:
                        san_move = self.board.san(move)
                        self.board.push(move)
                        self.move_history.append(san_move)
                        return True
                except ValueError as e:
                    logger.debug(f"Castling parse error: {e}")
                    pass

            # Try parsing as UCI (e2e4 format)
            try:
                move = chess.Move.from_uci(move_str)
                if move in self.board.legal_moves:
                    san_move = self.board.san(move)  # Convert to SAN for history
                    self.board.push(move)
                    self.move_history.append(san_move)
                    return True
            except ValueError:
                pass

            # Special handling for piece moves with ambiguous notation
            if move_str[0].isupper() and len(move_str) >= 3:
                piece_symbol = move_str[0].lower()
                if piece_symbol in chess.PIECE_SYMBOLS:
                    piece_type = chess.PIECE_SYMBOLS.index(piece_symbol)

                    # Extract destination square from the move string
                    dest_str = move_str[-2:]
                    try:
                        dest_square = chess.parse_square(dest_str)
                    except ValueError:
                        return False

                    # Find all pieces of the correct type that can move to the destination
                    valid_moves = []
                    for legal_move in self.board.legal_moves:
                        piece = self.board.piece_at(legal_move.from_square)
                        if (piece and
                                piece.piece_type == piece_type and
                                piece.color == self.board.turn and
                                legal_move.to_square == dest_square):
                            valid_moves.append(legal_move)

                    if len(valid_moves) == 1:
                        move = valid_moves[0]
                        san_move = self.board.san(move)
                        self.board.push(move)
                        self.move_history.append(san_move)
                        return True

            return False

        except (ValueError, AttributeError) as e:
            logger.error(f"Move error: {e}")
            return False

    def _handle_move(self, message: str, move: str) -> Dict:
        """Handle move intent with rich analysis"""
        # Special check for castling moves
        if move in ['e1-g1', 'e1-c1', 'e8-g8', 'e8-c8']:
            castling_move = 'O-O' if move in ['e1-g1', 'e8-g8'] else 'O-O-O'
            if not self._make_move(castling_move):
                return {
                    "status": "error",
                    "message": "Invalid castling move.",
                    "moves": self.move_history
                }
        # Check for same square move (click without dragging)
        elif len(move) >= 4 and move[:2] == move[2:4]:
            return {
                "status": "ignore",
                "message": "",
                "moves": self.move_history
            }
        # Handle regular moves
        elif not self._make_move(move):
            return {
                "status": "error",
                "message": "That's not a legal move. Try again!",
                "moves": self.move_history
            }

        # Assess the quality of the player's move
        player_move_quality = self._get_move_quality(self.board.move_stack[-1]) if self.board.move_stack else None

        # Check for game end after player's move
        game_end = self._check_game_end()
        if game_end["status"] == "game_over":
            self.game_in_progress = False
            return {
                "status": "success",
                "message": f"{move}. {game_end['message']}",
                "moves": self.move_history
            }

        # Get Maia's response
        try:
            if not self.maia_engine:
                return {
                    "status": "error",
                    "message": "Chess engine not available.",
                    "moves": self.move_history
                }

            maia_move = self.maia_engine.get_best_move(self.board)
            san_response = self.board.san(maia_move)
            self.board.push(maia_move)
            self.move_history.append(san_response)

            # Check for game end after Maia's move
            game_end = self._check_game_end()
            if game_end["status"] == "game_over":
                self.game_in_progress = False
                response_msg = f"{move}. I play {san_response}. {game_end['message']}"
                self.chat_history.append({"role": "user", "content": message})
                self.chat_history.append({"role": "assistant", "content": response_msg})
                return {
                    "status": "success",
                    "message": response_msg,
                    "moves": self.move_history
                }

            # Get rich position analysis for the LLM
            position_analysis = self._get_position_analysis()

            # For lone moves, still provide brief commentary with analysis
            prompt = self.prompt_maker.create_move_prompt(
                user_move=move,
                maia_move=san_response,
                move_history=self.move_history,
                chat_history=self.chat_history,
                user_message=message,
                position_analysis=position_analysis,
                move_quality=player_move_quality
            )

            analysis = model_manager.quick_response(prompt)
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": analysis})

            return {
                "status": "success",
                "message": analysis,
                "moves": self.move_history
            }

        except Exception as e:
            logger.error(f"Error in move handling: {e}")
            return {
                "status": "error",
                "message": "Something went wrong processing that move.",
                "moves": self.move_history
            }

    def _handle_explanation(self, message: str) -> Dict:
        """Handle explanation requests with comprehensive analysis"""
        try:
            # Get rich position analysis
            position_analysis = self._get_position_analysis()

            # Get top moves from engine
            top_moves_info = ""
            if self.maia_engine:
                top_moves = self.maia_engine.get_top_moves(self.board, num_moves=3)
                top_moves_info = self.prompt_maker.format_top_moves(top_moves)

            # Generate response using enhanced prompt
            prompt = self.prompt_maker.create_explanation_prompt(
                move_history=self.move_history,
                chat_history=self.chat_history,
                user_message=message,
                position_analysis=position_analysis,
                top_moves_info=top_moves_info
            )

            response = model_manager.quick_response(prompt)
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": response})

            return {
                "status": "success",
                "message": response,
                "moves": self.move_history
            }

        except Exception as e:
            logger.error(f"Error in explanation handling: {e}")
            # Fallback to simpler response
            return self._handle_chat(message)

    def _handle_chat(self, message: str) -> Dict:
        """Handle general chat with position context"""
        try:
            position_analysis = self._get_position_analysis()

            prompt = self.prompt_maker.create_chat_prompt(
                move_history=self.move_history,
                chat_history=self.chat_history,
                user_message=message,
                position_analysis=position_analysis
            )

            response = model_manager.quick_response(prompt)
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": response})

            return {
                "status": "success",
                "message": response,
                "moves": self.move_history
            }
        except Exception as e:
            logger.error(f"Error in chat handling: {e}")
            return {
                "status": "error",
                "message": "I'm having trouble responding right now.",
                "moves": self.move_history
            }

    def _check_game_end(self) -> Dict[str, str]:
        """Check if the game has ended and return appropriate message"""
        if self.board.is_game_over():
            outcome = self.board.outcome()
            if outcome.winner == chess.WHITE:
                return {
                    "status": "game_over",
                    "message": "Checkmate! Congratulations, you won! Want to play again?",
                    "moves": self.move_history
                }
            elif outcome.winner == chess.BLACK:
                return {
                    "status": "game_over",
                    "message": "Checkmate! I got you this time. Want a rematch?",
                    "moves": self.move_history
                }
            else:
                # Handle draws
                termination_messages = {
                    chess.Termination.STALEMATE: "Stalemate! It's a draw.",
                    chess.Termination.INSUFFICIENT_MATERIAL: "Draw - not enough pieces left to checkmate.",
                    chess.Termination.FIFTY_MOVES: "Draw by the fifty-move rule.",
                    chess.Termination.THREEFOLD_REPETITION: "Draw by threefold repetition.",
                }
                msg = termination_messages.get(outcome.termination, "It's a draw!")
                return {
                    "status": "game_over",
                    "message": f"{msg} Want to play again?",
                    "moves": self.move_history
                }

        return {
            "status": "",
            "message": "",
            "moves": self.move_history
        }

    def _reset_game(self):
        """Reset the game state"""
        self.board.reset()
        self.move_history.clear()
        self.chat_history.clear()
        self.game_in_progress = True
        self.player_color = chess.WHITE
        self._last_eval = None

    def _handle_unknown(self, message: str) -> Dict:
        """Handle unknown intents"""
        return {
            "status": "error",
            "message": "I didn't quite get that. You can make a move, ask a question about the position, or type 'play' to start a new game!",
            "moves": self.move_history
        }

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'maia_engine') and self.maia_engine:
            self.maia_engine.close()

    def __del__(self):
        self.close()
