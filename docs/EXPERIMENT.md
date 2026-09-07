# NEMESIS: single-player experiment

## Research question

Can a personal model of a player's estimated move regret identify challenging positions and steer an opponent toward them without a large sacrifice in search strength? Does practicing against that opponent reduce the player's subsequent errors on held-out positions?

These are two separate claims. Demonstrating changed move selection establishes neither predictive validity nor improved learning.

## Current implementation

NEMESIS learns online from the player's own moves. Its target is a bounded regret estimate from a chess search, not a known label of human weakness. The five UI contexts are interpretable positional features, not mutually exclusive or causal categories. The neural output estimates normalized regret, not a calibrated probability of losing or blundering.

There is one personal browser profile. Both modes update the same model; baseline mode disables its use in move selection. This makes baseline useful for software checks and exploratory play, but ordinary alternating live games alone do not isolate the treatment effect. The player learns, openings change, and the network evolves between games.

## Suggested study protocol

1. Establish a fixed evaluator and analysis budget. Prefer a strong UCI engine over the bundled two-ply demo. Keep evaluator versions and settings with exported results.
2. Gather initial play and reserve later games/positions for testing. Make splits at the game level to reduce leakage between adjacent positions.
3. For offline prediction, freeze a saved model and predict before revealing each held-out human move. Compare with a constant mean-regret predictor and context-frequency baselines. Track MAE and reliability bins for the normalized target, with sample counts.
4. For opponent behaviour, run adaptive and baseline policies from identical positions with the same frozen profile. Record the chosen move, estimated search cost, predicted human regret, and whether the move changes. Mate-band results must preserve the search choice.
5. For practice outcomes, predefine an alternating or randomized session schedule, time control, and an independent set of assessment positions. Measure errors before and after training; disclose practice/order effects and that a single subject cannot establish population-level effectiveness.
6. Export after sessions. Current JSON stores bounded history and replay; it is not an unlimited research archive. Keep engine labels with observations and separate fallback-engine data from strong-engine data.

## Useful reported measurements

| Question | Measurement |
| --- | --- |
| Does the personal model predict anything? | Held-out error versus constant/context baselines |
| Does adaptation influence play? | Changed choices / eligible positions |
| What does personalization cost? | Search-score difference between baseline and selected continuation |
| Is the context adequately sampled? | Number of relevant positions and estimated errors, by context |
| Does practice help this player? | Error change on independent assessment positions |
| Is it practical to play? | Median and tail response latency; engine fallback count |

The code currently includes a synthetic-label regression on a real board: training contrasting responses to positions following `...Nd7` and `...Qb6` changes the selected move while respecting the 65cp bound. It demonstrates that gradients reach the decision policy. It is not a human experiment.

## Next research work

Use stronger and deeper teaching labels; add held-out model evaluation tooling and profile snapshots; investigate richer board encodings or a pretrained player-policy prior; test uncertainty gating; collect real sessions. Do not tune and report on the same positions.
