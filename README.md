# Chess LLM Tutor

An interactive chess tutor that uses LLMs to provide real-time coaching. Built as my university thesis.

## How it works

- **Play chess** against a Maia engine (human-like neural network chess engine)
- **Chat naturally** — ask about positions, strategy, mistakes, or next moves
- **Intent classification** routes your message (analysis request, move command, general question)
- **LLM generates responses** grounded in the current board state via structured prompts

## Stack

- **Backend:** Django + python-chess + Maia engine
- **Frontend:** Chessboard.js + vanilla JS
- **LLM:** Claude via structured prompt pipelines

## Architecture

```
views.py          → API endpoints (chat, moves, board state)
ChessLogic.py     → Board management, move validation, Maia integration
PromptMaker.py    → Context-aware prompt construction from board state
intent.py         → Classifies user messages into action types
maia_engine.py    → Human-like chess engine for adaptive difficulty
```
