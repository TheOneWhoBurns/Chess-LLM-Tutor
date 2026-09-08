# NEMESIS experiment protocol

This project asks three separate questions: can a neural adapter predict one person's moves better than frozen Maia, do those predictions change the opponent's choices, and does practice against that opponent help the player? A changed move alone answers neither the first nor the third question.

## Implementation under study

Frozen Maia 1500 supplies probabilities over legal moves. The personal adapter learns from the human's chosen move using a bounded neural residual and regularization toward Maia. It does not learn a Stockfish-regret target or classify predefined weaknesses. Historical fitting and subsequent live updates are separate stages. The 96-observation replay limits live updates, not the imported training corpus. The `samples / (samples + 64)` adaptation schedule is a prototype choice, not a validated sample-size or confidence rule.

Stockfish 19 supplies independent chess scores. Candidate selection starts from up to six moves in an 80,000-node MultiPV search and retains those within 65cp of the best root result. Each candidate receives a 120,000-node MultiPV analysis of every legal human reply. Its refreshed value is the negative of the best human reply score, from NEMESIS's perspective. Terminal checkmate receives `MATE - 1`; a terminal draw receives zero.

A second eligibility check retains candidates within 65cp of the best refreshed value in the initial pool. It also excludes a newly discovered losing mate when another candidate is outside the losing-mate band. If the best refreshed result is itself in the positive or negative mate band, only its best mate value is eligible, preserving the quickest found win or longest found loss. The original root mate guard and baseline mode remain locked to the original root-best move.

Reply loss is the difference from the best human reply, capped at 2,000cp before taking its probability-weighted expectation. Both the frozen-Maia and personal choices maximize the refreshed candidate value plus expected human loss over the same eligible pool. The second search can therefore revise an earlier root score without that revision being attributed to personal learning.

Node budgets are shared across all variations in each search; they are not per-move budgets. Both finite-search scores and their 65cp limits can be inaccurate, and candidates outside the initial root pool are not recovered by the refreshed comparison. Record versions, settings, and returned search depth/nodes when interpreting results.

## Imported data and chronological evaluation

The importer reads the requested player's public Chess.com archive index and monthly game records. Requests are serial and older completed months are cached; the current month is refreshed. A manifest records archive hashes, accepted games, observed choices and skip reasons. Games are deduplicated by their API URL, validated by replaying the PGN, and sorted by game end time with URL as a stable tie-breaker. Standard chess is supported; malformed records, nonstandard variants and records that do not identify the requested player are excluded and counted. Public archives cannot supply unavailable or deleted games.

Only the requested player's choices become observations, using the color recorded in each game's API entry. Both White and Black games are usable. Split by whole game before generating examples; adjacent positions from the same game must not appear in both partitions. Preserve the manifest, accepted game identifiers, chronological boundary, settings and code revision with each reported result.

The evaluation checkpoint is trained on earlier games and frozen while both it and unchanged Maia predict the later held-out games. Held-out moves do not update that checkpoint. Mean negative log likelihood measures the probability assigned to actual choices; lower is better. Top-choice accuracy is the fraction whose most probable move matches the observation. Report paired metrics and the numbers of games and choices, including excluded data.

After recording the frozen comparison, fit a separate deployment model on all accepted games, including those previously held out. The application uses this all-games model and continues learning from live moves. The saved holdout report describes the earlier evaluation checkpoint; it is **not** an out-of-sample score for the deployed all-games weights. Once the later games are used for fitting or tuning, a new future assessment set is needed for another independent test. Do not choose settings using the reported holdout and then present it as untouched test data.

### Current offline runner

`python scripts/train_history.py --username YOUR_USERNAME --workers 4` runs the import, cached Maia inference, fitting and frozen evaluation. The default split uses the oldest 80% of accepted games for fitting and the newest 20% for evaluation, rounded to whole games; a one-game dataset has no held-out assessment. Each fit uses three fixed epochs with deterministic shuffled game order and visits every player choice each epoch. The deployment fit starts fresh and uses all accepted games for three epochs; it is not a continuation of training on the holdout checkpoint. Observation counts count each recorded choice once, not again on later epoch visits.

The offline optimizer updates all personal-adapter parameters with one gradient step per batch of up to 64 consecutive player choices within each shuffled game. It uses move-choice negative log likelihood, a `0.08` penalty on `KL(personal || Maia)`, and `0.0005` L2 weight regularization. The learning rate is `0.12`, with the total gradient norm clipped to `5`. The game-shuffle seed starts at `1701`, changes by epoch and uses a separate fixed offset for the deployment fit. Live learning uses the same objective and gradient optimizer but makes ten steps over a bounded replay batch of up to 24. These are fixed implementation settings, not values selected by the held-out result.

The Maia weights stay frozen and the target remains the player's actual move. This offline fit uses the full selected history; the application's small live replay buffer does not truncate that corpus. After each fit, the retained replay is restored to the latest 96 choices in chronological order for subsequent live learning. Runtime and source fingerprints identify cached priors and resumable checkpoints, including the Maia quantization setting and full position history. Keep these fingerprints with results, and treat `--max-games` as an explicitly limited smoke dataset. `--priors-only` prepares the cache without fitting or evaluating.

After prior caching completes, `--fit-only` verifies the completed local archive cache and resumes fitting from the last whole-game checkpoint without network requests or a full prior rescan. A missing required prior fails when requested; this mode does not substitute probabilities. It is mutually exclusive with `--priors-only` and preserves the existing split, optimizer state and source-compatibility checks.

By default, `.runtime/players/YOUR_USERNAME/training/` contains `split_model.json`, `deployment_model.json`, `report.json`, progress in `status.json`, and resume state in `checkpoint.json`. Preserve the split model with the report so the frozen comparison can be reproduced. Activate the all-games model with `python scripts/activate_player.py --username YOUR_USERNAME --training-dir .runtime/players/YOUR_USERNAME/training`; activation does not turn the earlier holdout report into an evaluation of the deployed weights. Replace `YOUR_USERNAME` with the lowercase account name in these commands and paths.

## Live prediction evaluation

For each human move, the application predicts with both frozen Maia and the current personal adapter **before** updating the adapter. It then records:

- `prior_probability` and `personal_probability`: probabilities assigned to the actual move.
- `prior_log_loss` and `personal_log_loss`: negative natural logarithms of those probabilities; lower is better.
- `prior_hit` and `personal_hit`: whether the most probable move matched the observation.
- `samples_before`, the position, the observed move, and the engine's separate move-quality estimate.

This predict-then-learn sequence is a prequential comparison: each prediction uses earlier observations, never the current move as training data. Compare paired losses on the same observations, show their evolution by game, and report the number of games and decisions. Top-prediction hit rate is a useful secondary measure; it discards most of the probability distribution.

Imported observations contribute to the policy's total `samples`. Live cumulative losses, hit counts and mean engine loss use `live_samples` as their denominator. Imported frozen-assessment metrics stay in the separate `training` report; never combine them with live predict-then-learn metrics or divide live hits by the imported-plus-live total.

Maia's probabilities are parsed from LC0's rounded policy output. Printed zeros receive a numerical floor, and the distribution is normalized. Probabilities are model estimates and are not known to be calibrated for this player. A one-person, correlated move sequence also limits the interpretation of uncertainty estimates; treat game-level results as more informative than pretending every adjacent position is independent.

## Opponent comparison

Compare three policies from identical positions with the same saved adapter and recorded engine settings:

| Policy | Choice rule |
| --- | --- |
| Stockfish baseline | Original root search's best move. |
| Frozen Maia | Refreshed score plus expected reply loss under Maia, among eligible candidates. |
| Personal NEMESIS | Refreshed score plus expected reply loss under the personal distribution, in the same eligible pool. |

Adaptive decision exports include all three chosen moves, candidate scores, expected losses, and the selected candidate's full reply distribution. `engine_score_cp`, `engine_cost_cp` and `baseline_move` retain the original root search values. `reply_score_cp`, `reply_cost_cp` and `reply_baseline_move` identify the refreshed comparison. All initially pooled candidates remain in the export, including rejected rows with `eligible: false`, a `guard_reason`, and `selected: false`.

`engine_changed` indicates departure from the original Stockfish root choice. `personal_changed` indicates departure from the frozen-Maia choice on the same eligible pool when the adapter has observations. Neither a Maia-only change nor a refreshed-search-only change is evidence that the adapter learned anything personal.

Report the personal-change rate among eligible adaptive decisions, the search-score cost of selected moves, expected loss under each distribution, and subsequently observed human loss. Keep mate-locked, terminal, and baseline decisions separate when choosing denominators. Expected loss is a hypothesis derived from the policy and engine, not an observed improvement in results. Full reply lists for unselected candidates are not retained in the export, so deeper counterfactual analysis requires rerunning or extending data collection.

The live baseline mode continues training the same adapter. Alternating ordinary games is useful for exploration but does not by itself isolate an effect: the player, model, openings, and positions all change over time.

## Practice outcomes and reproducibility

Before a training study, define a session schedule, assessment procedure, and success measure. Use independent assessment positions before and after practice, control access to hints, and keep game conditions consistent. Randomize or counterbalance practice conditions where feasible, and disclose order effects and the limitations of a single subject. The present app plays White without a clock; it does not record decision time.

Save a JSON export after every session, including policy snapshots and decision records. Retain the matching game PGNs, import manifest, offline training report, evaluation and deployment checkpoints, code revision, Stockfish and LC0 versions, Maia weights hash, backend, and engine settings. The setup uses pinned Stockfish and Maia artifacts; record the installed LC0 version as well.

Activation and each live move verify the saved prior fingerprint against the installed Maia weights, LC0 executable, backend, policy-wrapper code and quantization identity. The first successful move pins this identity for an initially untrained profile. A mismatch blocks the move rather than mixing predictions from different priors. Restore the matched runtime or begin a separately identified training run before comparing further results. Engine failures and watchdog timeouts do not save partial moves or observations.

Live profile history is bounded to the current game and 30 archived games, with 96 observations in neural replay. Cumulative live metrics may therefore cover observations older than the detailed profile export. The complete imported games and offline artifacts are kept separately under ignored runtime directories. Regular external archives are necessary to preserve a full study. Version-1 data is retained under `legacy_v1`; do not combine its old learning metrics with version-2 move-choice predictions.

Activating a player configures one shared local profile across browsers. Those browsers operate on the same model and game; they are not separate study participants or independent experimental conditions. Without an activated player, browser sessions identify separate profiles. Keep the local database and configured-player file with the study state.

Runtime and import tests establish implementation behavior. Repeated synthetic demonstrations establish that the adapter can learn preferences. A chronological held-out report can provide evidence about prediction on that particular period when its split and settings were fixed in advance. None of these alone demonstrates better opponent results or improved learning outcomes; those require the separate comparisons above.
