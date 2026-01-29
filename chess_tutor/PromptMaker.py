# PromptMaker.py
"""
Prompt generation for the chess tutor LLM.
Creates context-rich prompts that include position analysis, threats, and tactics.
"""

import re
from typing import List, Dict, Optional


class PromptMaker:
    def __init__(self):
        # Define move pattern for detecting lone moves
        self._move_pattern = re.compile(
            r'^[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?\+?\#?$|^O-O(-O)?$'
        )

    def _format_move_history(self, move_history: List[str]) -> str:
        """Format move history in standard chess notation"""
        if not move_history:
            return "No moves yet."
        return " ".join(
            f"{i//2 + 1}.{move}" if i % 2 == 0 else move
            for i, move in enumerate(move_history)
        )

    def _format_chat_history(self, chat_history: List[Dict[str, str]], limit: int = 6) -> str:
        """Format recent chat context"""
        recent = chat_history[-limit:] if len(chat_history) > limit else chat_history
        if not recent:
            return "No previous conversation."
        return "\n".join([
            f"{'User' if msg['role'] == 'user' else 'You'}: {msg['content']}"
            for msg in recent
        ])

    def create_move_prompt(self, user_move: str, maia_move: str,
                          move_history: List[str], chat_history: List[Dict[str, str]],
                          user_message: str, position_analysis: str = "",
                          move_quality: Optional[str] = None) -> str:
        """
        Create prompt for move analysis with rich position context.

        Args:
            user_move: The move made by the user
            maia_move: The response move by Maia (engine)
            move_history: List of all moves in the game
            chat_history: Previous chat messages
            user_message: The current message from the user
            position_analysis: Rich position analysis from ChessAnalyzer
            move_quality: Assessment of the user's move quality
        """
        moves_formatted = self._format_move_history(move_history)
        chat_formatted = self._format_chat_history(chat_history)

        # Build the context section
        context_parts = []

        if position_analysis:
            context_parts.append(position_analysis)

        if move_quality:
            context_parts.append(f"\n=== USER'S MOVE ASSESSMENT ===\n{move_quality}")

        analysis_context = "\n".join(context_parts) if context_parts else "No analysis available."

        return f"""You are a friendly chess tutor playing as Black against a student (White). You're having a casual, encouraging conversation while helping them improve.

=== GAME STATE ===
Moves so far: {moves_formatted}
User just played: {user_move}
Your response (as Black): {maia_move}

{analysis_context}

=== RECENT CHAT ===
{chat_formatted}

=== YOUR TASK ===
Respond naturally to the user's move. Your response should:

1. **React to the position**:
   - If their move was a BLUNDER or MISTAKE, gently point this out! This is critical for learning.
   - If there are HANGING PIECES or TACTICS, mention them (especially if it helps them learn).
   - If they made a GOOD or BRILLIANT move, acknowledge it!

2. **Comment on your (Maia's) response move** - what you're trying to achieve.

3. **Keep it conversational**:
   - Talk like a friend teaching chess, not a commentary bot
   - Use "I" for yourself (Black) and "you" for them (White)
   - Be encouraging but honest about mistakes
   - Use casual language, maybe an emoji here or there

4. **Be concise**: 1-3 sentences max for normal moves, slightly more if there's something important to teach.

User's message: "{user_message}"

Respond now:"""

    def create_explanation_prompt(self, move_history: List[str],
                                 chat_history: List[Dict[str, str]],
                                 user_message: str,
                                 position_analysis: str = "",
                                 top_moves_info: str = "") -> str:
        """
        Create a detailed prompt for chess explanations.

        Args:
            move_history: List of all moves in the game
            chat_history: Previous chat messages
            user_message: Current user question
            position_analysis: Rich analysis from ChessAnalyzer
            top_moves_info: Information about best moves from engine
        """
        moves_formatted = self._format_move_history(move_history)
        chat_formatted = self._format_chat_history(chat_history)

        engine_analysis = ""
        if top_moves_info:
            engine_analysis = f"\n=== ENGINE SUGGESTIONS ===\n{top_moves_info}"

        return f"""You are a friendly chess tutor helping a student understand the position. You're playing as Black.

=== GAME STATE ===
Moves played: {moves_formatted}

{position_analysis}
{engine_analysis}

=== RECENT CHAT ===
{chat_formatted}

=== YOUR TASK ===
The student is asking: "{user_message}"

Provide a helpful explanation that:
1. **Directly answers their question** using the position analysis above
2. **Teaches a concept** if relevant (tactics, strategy, common patterns)
3. **Hints at good moves** without giving away the exact best move (let them figure it out!)
4. **References specific details** from the analysis (threats, tactics, who's winning)

Keep it conversational and encouraging. Be specific - use square names, piece names, and concrete examples from the current position.

2-4 sentences is ideal. Don't overwhelm them.

Your response:"""

    def create_chat_prompt(self, move_history: List[str],
                          chat_history: List[Dict[str, str]],
                          user_message: str,
                          position_analysis: str = "") -> str:
        """
        Create prompt for general chess questions/chat.

        Args:
            move_history: List of all moves in the game
            chat_history: Previous chat messages
            user_message: Current user message
            position_analysis: Current position analysis (optional)
        """
        moves_formatted = self._format_move_history(move_history)
        chat_formatted = self._format_chat_history(chat_history)

        position_context = ""
        if position_analysis:
            position_context = f"\n=== CURRENT POSITION ===\n{position_analysis}\n"

        return f"""You are a friendly chess tutor playing as Black. The student wants to chat or ask a question.

=== GAME STATE ===
Moves played: {moves_formatted}
{position_context}
=== RECENT CHAT ===
{chat_formatted}

=== STUDENT'S MESSAGE ===
"{user_message}"

Respond helpfully and naturally. If they're asking about the current game, reference the position analysis. If it's a general chess question, share your knowledge. Keep it friendly and conversational!

Your response:"""

    def create_game_start_response(self) -> str:
        """Fixed response for new game"""
        return "Let's play! I'll be Black. Your move - show me what you've got! :)"

    def create_game_end_response(self, result: str) -> str:
        """Fixed response for game end"""
        responses = {
            "resign": "Good game! You played well. Want a rematch?",
            "checkmate": "Checkmate! Great game - want to play again?",
            "draw": "It's a draw! Solid play from both sides. Another game?",
        }
        return responses.get(result, "Game over! Would you like to play again?")

    def create_no_game_response(self) -> str:
        """Fixed response when no game is in progress"""
        return "No game in progress. Type 'play' to start a new game!"

    def _is_lone_move(self, message: str) -> bool:
        """Check if message is just a move notation without additional text"""
        return bool(self._move_pattern.match(message.strip()))

    def format_top_moves(self, top_moves: List[Dict]) -> str:
        """Format engine's top move suggestions for the prompt"""
        if not top_moves:
            return ""

        lines = ["Best moves in this position:"]
        for i, move_info in enumerate(top_moves[:3], 1):
            eval_pawns = move_info.get('evaluation', 0) / 100
            san = move_info.get('san', '?')
            mate = move_info.get('mate')

            if mate:
                eval_str = f"Mate in {abs(mate)}"
            else:
                eval_str = f"{eval_pawns:+.1f}"

            lines.append(f"  {i}. {san} (eval: {eval_str})")

        return "\n".join(lines)


# Global instance
prompt_maker = PromptMaker()
