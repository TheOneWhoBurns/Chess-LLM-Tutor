# NEMESIS

**Your habits. My advantage.** A personal chess opponent that learns where one player struggles and steers play toward those positions.

NEMESIS upgrades the original Chess LLM Tutor thesis into a runnable, observable opponent-modelling prototype. It has a real, online-trained neural player model, a playable chess arena, persistent memory across encounters, evidence-based weakness summaries, and a baseline comparison mode. No accounts or API keys are needed for the core game.

## Run locally

Requires Python 3.12 or newer with the supplied lockfile (tested on Python 3.14).

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
python manage.py migrate
python manage.py runserver 127.0.0.1:8765
```

Open http://127.0.0.1:8765. Play White by dragging, clicking a piece and its destination, or entering SAN/UCI moves such as `e4`, `Nf3`, `O-O`, or `e2e4`. Promotion offers all four pieces. Refreshing resumes the encounter. Starting another encounter retains learning and archives the previous game. “Forget my training data” clears this browser profile after confirmation.

`requirements.lock` records the versions tested for this upgrade. `requirements.txt` declares the compatible dependency ranges.

State lives in `nemesis.sqlite3`, linked to a year-long Django session cookie. Keep that database and the browser cookie to retain your profile; a different browser or cleared cookies starts a separate profile. This is a personal local app, with no identity/account system. The original tracked `db.sqlite3` is not used or modified.

## What learns

The player model is a NumPy MLP: **12 inputs → 16 tanh hidden units → 1 sigmoid output**. It starts from seeded random weights, not a pretrained chess policy.

1. Extract bounded features **before your move**: tactical pressure, king pressure, development, pawn structure, phase, material balance, attacked undefended material, remaining material phase, pawn count, legal-move count, check, and move number.
2. Evaluate every legal candidate from the same root and the same side's perspective. Estimated regret is `max(0, best_score - chosen_score)` in centipawns.
3. Train on the soft label `min(regret / 300, 1)` with 24 gradient steps over up to 128 recent observations. JSON-serialized weights and replay memory persist in the database.
4. After eight observations, rerank the opponent's candidate moves using the model's predicted regret in the resulting human-to-move position:

   `personal_score = search_score + 240 × influence × predicted_regret`

   Influence is zero for the first seven examples, then `min(samples / 40, 1)`. These are prototype hyperparameters, not calibrated confidence estimates.
5. Consider only moves within **65 centipawns** of the best search score. If the best score is in the mate band, keep the search's best continuation. This bound is relative to the configured search; it does not guarantee objective chess soundness.

**Baseline mode** still collects examples but disables the learned reranking. There is no hidden Elo, invented accuracy score, or predetermined weakness profile. Context summaries count estimated errors of at least 100cp and only flag a pattern after five relevant positions. A context association is not a causal diagnosis. Mean regret in profile summaries is capped at 1,000cp per observation; per-move export retains the uncapped estimate.

## Chess strength and research limitations

The default opponent uses a small, deterministic two-ply minimax search with material and positional heuristics. It runs without external engines, weights, or a cloud service. It is deliberately labelled **Demo search · 2 ply** in the UI. It can miss longer tactics and produce noisy teaching labels; the player network is not an independently trained chess engine.

For better teaching signals, connect an installed UCI engine (for example, Stockfish):

```sh
export NEMESIS_UCI_ENGINE=/absolute/path/to/stockfish
python manage.py runserver 127.0.0.1:8765
```

For an existing lc0/Maia setup, optionally pass a weights file:

```sh
export NEMESIS_UCI_ENGINE=/absolute/path/to/lc0
export NEMESIS_UCI_WEIGHTS=/absolute/path/to/maia-1100.pb.gz
```

UCI analysis requests all legal root moves with a 0.4-second budget. If the engine is missing, fails, or returns an incomplete candidate list, the app falls back to the demo search and exposes that in its status. The external UCI path is covered with mocks; a real external binary/weights pair was not exercised in this upgrade. Maia predicts human-like play, so its evaluations should not be treated as an unquestionable mistake oracle. The original Maia/LLM modules remain as legacy thesis reference, but are not imported on app startup. Their optional Transformers/Anthropic dependencies are not required by the new arena.

This implementation establishes the mechanism of personalization. It **does not yet demonstrate** improved win rate, accurate weakness classification, or better learning outcomes for a real player. Eight observations enable experimentation; they do not establish statistical validity.

## Thesis evaluation

Use the Field notes page to export JSON containing network state, current and up to 30 archived encounters, and move events (mode, move, regret, alternative, engine label, and whether adaptation changed the reply). Export the current game as PGN from the arena. Back up exports periodically; the archive is intentionally bounded.

For a credible single-player study, alternate adaptive and baseline encounters, use a stable strong evaluator, and reserve later positions/games for evaluation. Report prediction error on unseen moves, calibration, adaptation frequency, estimated strength sacrificed, and errors by context over time. Keep training/evaluation games separate; changing profiles or engines mid-comparison confounds the result. Replaying synthetic labels proves the mechanism works, not that human skill improves. See [docs/EXPERIMENT.md](docs/EXPERIMENT.md).

## Validation

```sh
python manage.py test chess_tutor
python manage.py check
python manage.py makemigrations --check --dry-run
```

Tests cover neural learning and serialization, a trained network changing a real chess continuation, cold-start gating, move-strength and mate constraints, legal game play, terminal positions, castling, underpromotion, persistence, browser isolation, request revisions, CSRF, exports, and transaction rollback on engine failure.

## Project map

- `chess_tutor/nemesis.py`: features, neural player model, search and move selection.
- `chess_tutor/views.py`: authoritative game actions, revision checks, profile persistence, PGN/JSON exports.
- `chess_tutor/models.py` and `migrations/`: player state and schema.
- `templates/chat.html`, `static/css/main.css`, `static/js/nemesis.js`: responsive arena, profile, and experiment notes.
- `chess_tutor/ChessLogic.py`, `intent.py`, `PromptMaker.py`, `legacy_models.py`, `maia_engine.py`: preserved original thesis implementation. The Maia evaluation sign error is fixed.

Django defaults to local development. For deployment, set `DJANGO_DEBUG=0`, `DJANGO_SECRET_KEY`, and `DJANGO_ALLOWED_HOSTS`, and configure HTTPS/static serving. Environment variables are read from the process; `.env` is a reference file, not automatically loaded. The old hardcoded service token was removed from settings; if it was real, revoke/rotate it because it still exists in Git history. No external service is called by the default application.

Technical references: [python-chess engine API](https://python-chess.readthedocs.io/en/latest/engine.html), [Django transactions](https://docs.djangoproject.com/en/5.2/topics/db/transactions/).
