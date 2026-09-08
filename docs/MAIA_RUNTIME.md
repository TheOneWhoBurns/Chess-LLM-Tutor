# Local Maia runtime

NEMESIS uses the released **Maia-1500** neural network as a frozen prior over human moves. It is a 6-block, 64-filter SE convolutional network with a classical 112-plane input, convolutional policy head, and WDL value head. A separate player adapter learns from the user's choices; this wrapper does not retrain Maia's pretrained weights. Its probabilities predict moves, not objectively best play. Stockfish supplies the independent move-quality evaluations.

## Reproduce the local setup

Install LC0 using the [upstream instructions](https://lczero.org/play/quickstart/). The validated local installation is `/opt/homebrew/bin/lc0`, reporting `Lc0 v0.32.1+git.dirty`, with the `blas` backend using Apple vecLib. Python uses the project's existing `python-chess` dependency; PyTorch, CUDA, accounts, and API keys are not required.

From the project root, download the official release asset:

```sh
mkdir -p .runtime/maia
curl --fail --location --output .runtime/maia/maia-1500.pb.gz \
  https://github.com/CSSLab/maia-chess/releases/download/v1.0/maia-1500.pb.gz
shasum -a 256 .runtime/maia/maia-1500.pb.gz
```

The downloaded file is runtime data and should remain outside Git. Optional environment settings are `NEMESIS_LC0_ENGINE` (executable path), `NEMESIS_MAIA_WEIGHTS` (weights path), and `NEMESIS_MAIA_BACKEND` (default `blas`; `eigen` is another CPU option when provided by the installed LC0 build).

## Download provenance

| Field | Recorded value |
|---|---|
| Source | [CSSLab/maia-chess release v1.0](https://github.com/CSSLab/maia-chess/releases/tag/v1.0) |
| Asset | `maia-1500.pb.gz`, GitHub asset ID `30665578` |
| Release date | 2021-01-14 |
| Retrieved | 2026-09-07 |
| File bytes | 1,258,199 |
| SHA-256 | `35ab6f20421d59e1df3b17c5a5016947af4c6761368ef84044a9a9c7619a9a00` |

The [official release API](https://api.github.com/repos/CSSLab/maia-chess/releases/tags/v1.0) reports the matching asset size and a null `digest`; no publisher checksum is available there. The hash above was computed from the HTTPS download and records the tested artifact. It is not a publisher-signed checksum. Gzip integrity and LC0's network decoder were checked successfully.

## Inference contract and validation

`MaiaPolicy.probabilities(board)` lazily starts LC0 and returns every legal UCI move with a positive, normalized probability. The process stays loaded between requests. The caller must serialize access and call `close()` during shutdown.

Inference uses classic search at `nodes=1`, policy temperature 1, one thread, and minibatch size 1. A new game token clears the search tree for each query while passing the board's actual move history. This follows [Maia's single-node inference instructions](https://github.com/CSSLab/maia-chess#how-to-run-maia). The wrapper streams each root `VerboseMoveStats` probability, as in the project's [original Python wrapper](https://github.com/CSSLab/maia-chess/blob/master/move_prediction/maia_chess_backend/uci.py). LC0's aggregate score is unused.

Verbose percentages have two decimal places. A printed `0.00%` represents an unknown probability below `0.00005`. The parser substitutes that interval's midpoint, `0.000025`, only for printed zeros, then renormalizes all moves and converts castling from king-to-rook encoding. This quantization-aware estimate lets the adapter learn unusual observed choices; it is not the original unrounded neural output. The cache fingerprint must include `lc0-percent-midpoint-v2`. Missing moves, illegal moves, invalid probability mass, missing files, and runtime failures raise `MaiaUnavailable`. There is no substitute model. An external 15-second watchdog closes a stalled process; the UCI startup handshake also has a timeout.

Run the protocol tests and real network probes:

```sh
NEMESIS_TEST_REAL_MAIA=1 .venv/bin/python -m unittest chess_tutor.test_maia_policy -v
```

All 12 tests passed locally, including real starting-position, both-color castling, and promotion probes, full legal-move coverage, parser corruption handling, and a mocked hung-process deadline. Initial startup plus inference took about 0.13 seconds; the next castling query took about 0.015 seconds. These are local smoke measurements, not a performance benchmark or evidence of successful personalization.

## Attribution and licensing

Maia is by CSSLab and collaborators: McIlroy-Young, Sen, Kleinberg, and Anderson, *Aligning Superhuman AI with Human Behavior: Chess as a Model System* (KDD 2020). See [the paper and repository](https://github.com/CSSLab/maia-chess). Maia's repository is distributed under [GPL-3.0](https://github.com/CSSLab/maia-chess/blob/master/LICENSE). LC0 is distributed under [GPL-3.0 with its documented additional permission](https://github.com/LeelaChessZero/lc0/blob/master/COPYING). Runtime downloads preserve the upstream sources and attribution here; they are not claimed as NEMESIS-trained models.
