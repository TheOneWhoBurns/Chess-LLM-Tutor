# NEMESIS

A local chess opponent for one player. NEMESIS learns which moves you tend to choose, then uses those predictions to select positions where your likely replies cost you material or position.

The application combines **Stockfish 19**, the pretrained **Maia 1500** human-move network, and a small neural adapter trained on your actual moves. It can import your public Chess.com history, evaluate personal predictions on later held-out games, and continue learning as you play locally. An optional **Astra coach** restores conversational help alongside the board. Live predictions are recorded before each update. Better prediction, different opponent choices, and more effective practice are separate experimental questions.

## Run locally

Use Python 3.12 or newer; the supplied lockfile was tested with Python 3.14. Install [LC0](https://lczero.org/play/quickstart/) separately and make its `lc0` executable available on your PATH. From the project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python scripts/setup_engines.py
python manage.py migrate
python manage.py runserver 127.0.0.1:8765
```

The setup script downloads the pinned macOS Stockfish binary and Maia weights into `.runtime/`. It does not install LC0. On other platforms, install compatible engines and set the paths below. `requirements.lock` pins the tested Python dependencies; `requirements.txt` declares their supported ranges.

Open [NEMESIS locally](http://127.0.0.1:8765). You play White without a clock. Drag a piece, select its source and destination, or enter a move such as `e4`, `Nf3`, `O-O`, or `e2e4`. Promotion supports all four pieces. Refreshing resumes the game; starting a new game retains your player model.

Optional runtime settings:

| Variable | Purpose |
| --- | --- |
| `NEMESIS_STOCKFISH` | Path to a Stockfish executable; otherwise use PATH or the downloaded macOS binary. |
| `NEMESIS_LC0_ENGINE` | Path to LC0; otherwise use `lc0` from PATH. |
| `NEMESIS_MAIA_WEIGHTS` | Maia weights file; default `.runtime/maia/maia-1500.pb.gz`. |
| `NEMESIS_MAIA_BACKEND` | LC0 inference backend; default `blas`, subject to the installed build. |
| `OPENAI_API_KEY` | Optional server-side key for the Astra coach; requires access to `gpt-6-astra`. |

Both chess engines run locally and stay loaded between moves. Chess play and personal-model training need no API key or remote inference service. Missing engines, incomplete policy output, or failed analysis return an error without saving a partial move or training update. See [the Maia runtime notes](docs/MAIA_RUNTIME.md) for inference settings, download provenance, and attribution.

## Astra chat

Open **Chat** beside the board to ask about your last mistake, NEMESIS's decision, or what to practice. Set `OPENAI_API_KEY` in the Django server's environment and restart the server to enable it. `.env.example` documents the settings; `.env` files are not loaded automatically. The key stays on the server. Configuration detection does not verify account access: an invalid key, unavailable model, or exhausted quota produces an explicit chat error, with no substitute model. The local opponent remains usable without chat.

The coach calls OpenAI's [Responses API](https://developers.openai.com/api/docs/guides/text) with `gpt-6-astra`, low reasoning effort, and `store: false`. Each request sends your message, recent conversation, the current board and move history, selected personal-model predictions, recent recorded mistakes, aggregate training results, and fresh Stockfish analysis when available. Neural weights, replay buffers, and the downloaded game archive are excluded from this coaching context. Completed conversations and their evidence snapshots are saved in the local database. JSON exports include the transcript, model identifiers, and game/position labels.

Chat explains evidence and offers practice advice; it does not play moves, modify the personal model, or train on the conversation. Historical move prediction alone cannot establish a recurring chess weakness, and the coach is instructed to distinguish predicted replies from mistakes you actually made. Control access to coaching when comparing practice outcomes.

## Import and train on Chess.com history

After installing the runtime, replace `YOUR_USERNAME` below with your lowercase Chess.com username:

```sh
python scripts/train_history.py --username YOUR_USERNAME --workers 4
python scripts/activate_player.py --username YOUR_USERNAME --training-dir .runtime/players/YOUR_USERNAME/training
```

The training runner imports all available public monthly archives for that player using Chess.com's read-only API. It accepts valid standard-chess games, deduplicates game URLs and learns only from that player's moves, whether playing White or Black. The import manifest records skipped records and their reasons. No Chess.com login is used, and unavailable or deleted games cannot be recovered from the public API.

The first run computes actual Maia probabilities for every accepted choice, using each position's full game history. `--workers 4` runs four persistent LC0 workers. Priors are cached on disk; completed older monthly downloads are reused, while the current month is refreshed. Repeating the command resumes compatible work. Source, runtime and model fingerprints prevent incompatible checkpoints from being silently reused. If those inputs change, choose a fresh training directory with `--output-dir`; existing checkpoints are not overwritten to force a match.

Historical fitting uses a fixed three epochs with deterministic shuffled game order and batches of up to 64 choices from each game. Each batch applies one gradient step to all personal-adapter parameters using the same objective as live learning. The earlier 80% of games train an evaluation model; the later 20% are evaluated with that model frozen. After recording the comparison against unchanged Maia, a fresh model receives a separate three-epoch fit on all accepted games for deployment. Every player choice in the selected games is used each epoch; `samples` counts each recorded choice once rather than counting repeated epoch visits. No settings are selected from the held-out scores.

The default player directory is `.runtime/players/YOUR_USERNAME/`:

| Artifact | Purpose |
| --- | --- |
| `priors.sqlite3` | Cached Maia move distributions. |
| `training/status.json` | Training progress. |
| `training/checkpoint.json` | Compatible resumable training state. |
| `training/split_model.json` | Frozen model fitted only to the earlier games. |
| `training/deployment_model.json` | Separate model fitted to all accepted games. |
| `training/report.json` | Dataset counts, settings and the frozen holdout comparison. |

`--priors-only` fills the Maia cache without fitting a model. `--max-games` limits a smoke run; omit it to use the complete accepted history. The activation command validates and loads the completed all-history deployment model into the local profile. It refuses a partial smoke fit or mismatched artifacts. Reactivating the same artifact preserves subsequent live games and learning; activating a different model over an existing named profile is refused. Reload the application to see the account identity, imported counts and chronological holdout report under **Player model**.

Once the import and Maia cache are complete, resume fitting without network access or another full prior scan:

```sh
python scripts/train_history.py --username YOUR_USERNAME --fit-only
```

`--fit-only` verifies the completed local archive cache and resumes from the last completed game checkpoint. It fails if a required prior is missing rather than replacing it with an estimate. It cannot be combined with `--priors-only`.

The saved policy records its Maia runtime identity: weights, LC0 executable, inference backend, policy-wrapper code and quantization version. Activation and every live move verify that identity. A fresh live profile records it on its first move. A mismatch stops play before the move or learning update is saved; restore the matching runtime, retrain, or reset the active player model to start again. Reset remains available when the engine is unavailable.

The reported holdout scores belong to `split_model.json`. They do not assess `deployment_model.json` out of sample, because its final fit includes those later games. A higher personal log loss is a worse result and is displayed as such. Subsequent live accuracy is measured separately, before learning from each new move.

## What learns

Maia supplies a probability for every legal human move. Its pretrained weights remain frozen. A separate NumPy network learns a residual adjustment from your demonstrated choices:

```text
personal probability = softmax(log(Maia probability) + adaptation weight × residual)
```

The adapter has 222 inputs, 24 tanh hidden units, and one bounded residual per candidate move. Inputs describe the board, source and destination squares, moving and captured pieces, promotion, checks, castling, en passant, and positional context. The zero-initialized output layer starts with exactly Maia's normalized probabilities.

Live training minimizes move-choice log loss with a penalty for departing from the Maia prior. It keeps the latest 96 observations and trains deterministic batches of up to 24. Each replay observation contains the position, supplied Maia probabilities, and the move you made. Imported history is fitted separately before deployment; the 96-item live replay is not the limit on how many historical choices can be used. Stockfish's evaluation is recorded separately and is not a neural training label.

The residual is bounded between -2 and 2. Its weight follows `samples / (samples + 64)`. This is a conservative adaptation schedule, not an estimate of confidence or proof that enough games have been played.

Before each update, NEMESIS records both models' probability of your actual move, their top prediction, and their log loss. Lower log loss means the model assigned more probability to the observed choice. These measurements compare personal predictions with frozen Maia on the same sequence of moves.

## How the opponent chooses

Stockfish evaluates up to six candidate moves with an 80,000-node MultiPV search. The initial pool contains moves within 65 centipawns of that search's best result. For each pooled candidate, Maia and the personal adapter predict every legal human reply; Stockfish evaluates those replies with a 120,000-node MultiPV search.

The later reply search refreshes each candidate's value to the negative of the best human reply's score. NEMESIS applies a second 65cp limit against the best refreshed value within the initial pool. A candidate rejected by this second check cannot be selected, even if the earlier root search rated it highly. A discovered losing mate is excluded when another candidate avoids that mate band. If the best refreshed value itself is a mate score, only candidates matching that best mate value remain eligible.

Each reply's estimated loss is its difference from the best human reply, capped at 2,000 centipawns. The adaptive opponent chooses the candidate maximizing:

```text
refreshed score for NEMESIS + sum(personal reply probability × reply loss)
```

These are node budgets per search, shared across its variations. Both 65cp limits are relative to finite searches within the initial candidate pool, not guarantees of objective soundness. If the best original root result is in the mate band, NEMESIS keeps Stockfish's original choice. This is one human-reply expectation layered on engine search; it does not simulate an entire future game with the personal model.

The decision record distinguishes the original Stockfish root choice, the frozen-Maia choice, and the personal choice. Both model choices use the same refreshed scores and eligible pool. `engine_changed` means the selected move differs from the original Stockfish choice; `personal_changed` means it differs from the frozen-Maia choice and there are personal observations. A change caused only by the refreshed search is not counted as a personal change. The interface labels the original search cost separately and exposes refreshed candidate scores, costs and exclusion reasons.

**Baseline mode** plays Stockfish's best move while continuing to collect observations and update the personal model. Switching modes does not create independent experimental groups.

## Saved data and experiments

Live state is saved in `nemesis.sqlite3`. Before a player is configured, a year-long browser session cookie identifies the profile, so a different browser or cleared cookies starts a separate profile. Activating an imported player writes `.runtime/player.json`: browsers using this local installation then share that configured player's model and active game. This is a one-person application with no account interface. The original tracked `db.sqlite3` is not used by this application.

Export the current game as PGN or export the saved profile as JSON. JSON includes policy weights, bounded replay, cumulative live prediction metrics, imported training metadata when present, current events and decisions, up to 30 archived games, and completed coaching transcripts with model identifiers and position labels. It is not a replacement for the complete downloaded history and offline training artifacts. Export periodically for a study.

The database, downloaded games, import manifests, policy caches, checkpoints and local player configuration are runtime data excluded from Git. Resetting the player model clears its learned policy, saved games and current game while keeping the configured account identity; it does not delete downloaded archives or training files under `.runtime/`.

Profiles from the earlier version retain their active board, game counts, mode, result, and archived games. Their full former state is preserved under `legacy_v1`. The incompatible old neural weights are not reused: version 2 starts a new personal move policy and new prediction metrics.

See [the experiment protocol](docs/EXPERIMENT.md) for prediction comparisons, controlled opponent comparisons, and a separate assessment of whether practice helps this player. The current implementation does not establish improved win rate, calibrated probabilities, or better learning outcomes.

## Validation and source

```sh
NEMESIS_TEST_REAL_MAIA=1 NEMESIS_TEST_REAL_ENGINES=1 python manage.py test chess_tutor
python manage.py check
python manage.py makemigrations --check --dry-run
```

Checks cover legal neural probabilities, learning from demonstrated moves, pre-update metrics, serialization, move selection and score perspectives, cost and mate constraints, special moves, persistence, migration, and rollback on engine failure. Import checks cover validation, deduplication, chronological ordering and resumable archive handling. Chat tests use mocked provider responses to verify context filtering, response handling, persistence and request conflicts; they do not establish live Astra access. The optional real-engine tests require the installed engine dependencies. Run the commands above for the current test results; passing implementation checks does not establish effective personalization.

The active opponent is implemented in `chess_tutor/maia_policy.py`, `player_policy.py`, `opponent.py`, and `views.py`. Public history import is in `chess_tutor/chesscom_import.py`; coaching is in `astra.py` and `chat_views.py`. The board and chat interface is in `templates/chat.html`, `static/css/main.css`, and `static/js/nemesis.js`. Remaining original tutor modules are not the active opponent.

Stockfish, LC0, and Maia are upstream projects; NEMESIS does not claim their pretrained work as its own. Runtime downloads stay outside Git. This repository defaults to a local Django development server.
