#!/usr/bin/env python3
"""Cache real Maia priors and train/evaluate one player's chronological history.

All positions from a game stay in the same 80/20 chronological partition.
Held-out predictions are made with a frozen training-only model. The separate
deployment model is subsequently fitted on the held-out choices as well.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import math
import os
from pathlib import Path
import re
import random
import sqlite3
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess

from chess_tutor.chesscom_import import (
    GameSkipped, ImportResult, fetch_games, iter_training_examples, parse_game,
)
from chess_tutor.maia_policy import MaiaPolicy
from chess_tutor.player_policy import PersonalPolicy

CACHE_VERSION = 2
RUN_VERSION = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def username_key(username):
    if not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", username):
        raise ValueError("Provide a valid Chess.com username.")
    return username.casefold()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".training-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as output:
            # dumps uses CPython's C encoder; bytes match the streaming encoder
            # while avoiding thousands of tiny writes per model checkpoint.
            output.write(json.dumps(value, sort_keys=True, allow_nan=False))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def maia_fingerprint():
    """Include weights, engine build, wrapper settings and quantization semantics."""
    from chess_tutor.runtime_identity import current_maia_fingerprint
    return current_maia_fingerprint()


class PriorCache:
    """On-disk complete legal distributions, namespaced by verified runtime."""
    def __init__(self, path, username, policy_fingerprint):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS priors_v2 (key TEXT PRIMARY KEY, probabilities TEXT NOT NULL)")
        username = username_key(username)
        row = self.connection.execute("SELECT value FROM metadata WHERE name='username'").fetchone()
        if row and row[0] != username:
            self.connection.close()
            raise ValueError("This prior cache belongs to another username.")
        self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('username', ?)", (username,))
        self.connection.commit()
        self.fingerprint = policy_fingerprint
        self.hits = self.misses = self.writes = 0

    def key(self, board):
        return digest({"version": CACHE_VERSION, "policy": self.fingerprint,
                       "root": board.root().fen(en_passant="fen"),
                       "history": [move.uci() for move in board.move_stack],
                       "fen": board.fen(en_passant="fen")})

    def get(self, board):
        row = self.connection.execute("SELECT probabilities FROM priors_v2 WHERE key=?", (self.key(board),)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        result = json.loads(row[0])
        self.validate(board, result)
        return result

    @staticmethod
    def validate(board, probabilities):
        legal = {move.uci() for move in board.legal_moves}
        if not isinstance(probabilities, dict) or set(probabilities) != legal or not legal:
            raise ValueError("A cached prior must contain every legal move.")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-7):
            raise ValueError("A cached prior has invalid probabilities.")

    def put(self, board, probabilities):
        self.validate(board, probabilities)
        self.connection.execute("INSERT OR REPLACE INTO priors_v2 VALUES (?, ?)",
                                (self.key(board), canonical(probabilities)))
        self.writes += 1
        if self.writes % 32 == 0:
            self.connection.commit()

    def close(self):
        self.connection.commit()
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def selected_games(games, username, max_games=None):
    username = username_key(username)
    if max_games is not None and (type(max_games) is not int or max_games < 1):
        raise ValueError("--max-games must be positive; omit it to use all games.")
    ordered = sorted(games, key=lambda row: (row["timestamp"], row["url"]))
    if len({game["url"] for game in ordered}) != len(ordered):
        raise ValueError("Source games contain duplicate URLs.")
    for game in ordered:
        color = game.get("user_color")
        if color not in ("white", "black"):
            raise ValueError("Source games must identify the player's color.")
        player = game.get(color)
        if isinstance(player, dict) and str(player.get("username", "")).casefold() != username:
            raise ValueError("Source game belongs to another username.")
    return ordered[:max_games] if max_games is not None else ordered


def dataset_fingerprint(games):
    return digest([{"url": game["url"], "timestamp": game["timestamp"],
                    "pgn": hashlib.sha256(game["pgn"].encode()).hexdigest(),
                    "user_color": game["user_color"], "time_class": game.get("time_class")}
                   for game in games])


def cached_games(username, cache_dir):
    """Reconstruct a completed import locally; no archive or prior rescan online."""
    username = username_key(username)
    directory = Path(cache_dir) / username
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("username") != username or manifest.get("status") != "complete" or \
            manifest.get("archives_completed") != manifest.get("archive_count"):
        raise ValueError("Fit-only requires a completed archive import for this username.")
    games, seen = [], set()
    for archive in manifest["archives"]:
        name = archive.get("file", "")
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}\.json", name):
            raise ValueError("The completed import contains an invalid archive filename.")
        payload = (directory / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != archive.get("sha256"):
            raise ValueError("A cached archive changed after import. Refresh the import before fitting.")
        for record in json.loads(payload)["games"]:
            url = record.get("url") if isinstance(record, dict) else None
            if url:
                if url in seen:
                    continue
                seen.add(url)
            try:
                games.append(parse_game(record, username))
            except GameSkipped:
                continue
    games.sort(key=lambda game: (game["timestamp"], game["url"]))
    if len(games) != manifest.get("accepted_games") or sum(game["human_moves"] for game in games) != manifest.get("human_moves"):
        raise ValueError("Cached game counts do not match the completed import.")
    return ImportResult(games=games, manifest=manifest, cache_dir=directory)


class Progress:
    def __init__(self, directory, username):
        self.path = Path(directory) / "status.json"
        self.username = username
        self.started = time.time()
        self.last_print = 0

    def write(self, phase, force=False, **values):
        elapsed = time.time() - self.started
        state = {"username": self.username, "phase": phase, "updated_at": time.time(),
                 "elapsed_seconds": round(elapsed, 2), **values}
        atomic_json(self.path, state)
        count = values.get("games_done", values.get("games_scanned", 0))
        if force or time.time() - self.last_print >= 5 or count and count % 50 == 0:
            print(canonical(state), flush=True)
            self.last_print = time.time()


def game_examples(game, iterator=iter_training_examples):
    # Materialize only one game so parser failures cannot produce partial training.
    examples = list(iterator(game))
    if not examples:
        raise ValueError("Game contains no player choices.")
    for example in examples:
        board, move = example["board"], example["move_uci"]
        if not board.is_valid() or chess.Move.from_uci(move) not in board.legal_moves:
            raise ValueError("History iterator returned an invalid demonstrated move.")
        if board.turn != (game["user_color"] == "white"):
            raise ValueError("History iterator returned the opponent's move.")
    return examples


def precompute_priors(games, cache, output_dir, username, workers=4,
                      provider_factory=MaiaPolicy, iterator=iter_training_examples):
    """Bounded parallel inference; only the coordinator accesses SQLite."""
    if not 1 <= workers <= 8:
        raise ValueError("Use between one and eight Maia workers.")
    progress = Progress(output_dir, username)
    local = threading.local()
    providers, pending, in_flight = [], {}, set()
    lock = threading.Lock()
    seen = games_done = 0
    skipped = []

    def predict(board):
        if not hasattr(local, "provider"):
            local.provider = provider_factory()
            with lock:
                providers.append(local.provider)
        return local.provider.probabilities(board)

    def collect(block):
        if not pending:
            return
        completed, _ = wait(pending, timeout=None if block else 0, return_when=FIRST_COMPLETED)
        for future in completed:
            board, key = pending.pop(future)
            in_flight.remove(key)
            cache.put(board, future.result())

    progress.write("priors", force=True, games_scanned=0, total_games=len(games), workers=workers)
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for game in games:
                try:
                    examples = game_examples(game, iterator)
                except ValueError as error:
                    skipped.append({"url": game["url"], "reason": str(error)})
                    examples = []
                for example in examples:
                    board = example["board"]
                    seen += 1
                    if cache.get(board) is not None:
                        continue
                    key = cache.key(board)
                    if key not in in_flight:
                        in_flight.add(key)
                        pending[executor.submit(predict, board)] = (board, key)
                    if len(pending) >= workers * 16:
                        collect(True)
                games_done += 1
                collect(False)
                progress.write("priors", games_scanned=games_done, total_games=len(games),
                               observations_seen=seen, cached_priors=cache.writes,
                               cache_hits=cache.hits, pending=len(pending), workers=workers,
                               skipped_games=len(skipped))
            while pending:
                collect(True)
        cache.connection.commit()
        result = {"games_scanned": games_done, "total_games": len(games), "observations_seen": seen,
                  "cached_priors": cache.writes, "cache_hits": cache.hits, "skipped": skipped}
        progress.write("priors_complete", force=True, **result)
        return result
    except BaseException as error:
        progress.write("failed", force=True, stage="priors", error=str(error), games_scanned=games_done,
                       observations_seen=seen, cached_priors=cache.writes)
        raise
    finally:
        for provider in providers:
            provider.close()


def empty_metrics():
    return {"observations": 0, "prior_log_loss_sum": 0., "personal_log_loss_sum": 0.,
            "prior_hits": 0, "personal_hits": 0, "prior_brier_sum": 0., "personal_brier_sum": 0.}


def add_metrics(metrics, prior, personal, chosen):
    metrics["observations"] += 1
    for name, distribution in (("prior", prior), ("personal", personal)):
        metrics[name + "_log_loss_sum"] -= math.log(distribution[chosen])
        metrics[name + "_hits"] += int(max(sorted(distribution), key=distribution.get) == chosen)
        metrics[name + "_brier_sum"] += math.fsum(
            (probability - int(move == chosen)) ** 2 for move, probability in distribution.items())


def summarize(metrics):
    count = metrics["observations"]
    result = {"observations": count}
    for name in ("prior", "personal"):
        for field, denominator in (("log_loss", "_log_loss_sum"), ("top1", "_hits"), ("brier", "_brier_sum")):
            result[name + "_" + field] = metrics[name + denominator] / count if count else None
    result["personal_log_loss_improvement"] = (
        result["prior_log_loss"] - result["personal_log_loss"] if count else None)
    return result


def run_training(games, username, output_dir, prior_provider, *, iterator=iter_training_examples,
                 policy_fingerprint, max_games=None, source_manifest=None):
    """Fixed three-epoch game-shuffled fits with a separate frozen holdout.

    No held-out result changes the optimizer or epoch count. Deployment starts
    fresh and fits all selected games; reported losses remain those of the
    saved 80%-only model. Counts represent observations, not repeated epochs.
    """
    username = username_key(username)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    selected = selected_games(games, username, max_games)
    if not selected:
        raise ValueError("There are no eligible games to train.")
    count = len(selected)
    split = max(1, min(count - 1, int(count * .8))) if count >= 2 else count
    identity = {"version": RUN_VERSION, "username": username,
                "dataset_fingerprint": dataset_fingerprint(selected),
                "policy_fingerprint": policy_fingerprint,
                "adapter_code_sha256": file_digest(ROOT / "chess_tutor/player_policy.py"),
                "source_games": len(games), "selected_games": count, "train_games": split,
                "heldout_games": count - split, "complete_dataset": count == len(games),
                "fit_method": "game_shuffled_minibatch_v1", "fit_epochs": 3,
                "batch_size": 64, "shuffle_seed": 1701}
    checkpoint_path = directory / "checkpoint.json"
    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text())
        if state.get("identity") != identity:
            raise ValueError("Existing checkpoint has a different username, source games, runtime or adapter. Use a new output directory; it was not overwritten.")
        policy = PersonalPolicy(state["policy"])
    else:
        policy = PersonalPolicy()
        state = {"identity": identity, "phase": "train", "epoch": 0, "cursor": 0, "policy": policy.dump(),
                 "heldout_metrics": empty_metrics(), "by_time_class": {}, "skipped": [],
                 "training_moves": 0, "deployment_moves": 0, "train_games_ok": 0,
                 "test_games_ok": 0, "deploy_games_ok": 0, "last_completed_game": None}
        atomic_json(checkpoint_path, state)
    progress = Progress(directory, username)

    def save():
        state["policy"] = policy.dump()
        atomic_json(checkpoint_path, state)
        progress.write(state["phase"], games_done=state["cursor"], total_games=count,
                       epoch=state["epoch"] + 1, epochs=identity["fit_epochs"],
                       samples=policy.samples, training_moves=state["training_moves"],
                       heldout_observations=state["heldout_metrics"]["observations"],
                       last_completed_game=state["last_completed_game"])

    def snapshot(path, purpose, summary=None):
        atomic_json(directory / path, {**identity, "purpose": purpose, "summary": summary,
                                      "samples": policy.samples, "policy": policy.dump(),
                                      "skipped": state["skipped"]})

    def recent_replay(indices):
        recent = deque(maxlen=min(policy.samples, 96))
        for index in reversed(list(indices)):
            try:
                examples = game_examples(selected[index], iterator)
            except ValueError:
                continue
            for example in reversed(examples):
                recent.appendleft({"board": example["board"], "chosen_uci": example["move_uci"],
                                   "prior": prior_provider(example["board"])})
                if len(recent) == min(policy.samples, 96):
                    policy.replace_replay(list(recent))
                    return
        policy.replace_replay(list(recent))

    def summary(trained_at=None):
        measured = summarize(state["heldout_metrics"])
        return {"username": username,
                "games_total": state["train_games_ok"] + state["test_games_ok"],
                "train_games": state["train_games_ok"], "test_games": state["test_games_ok"],
                "train_positions": state["training_moves"],
                "test_positions": measured["observations"],
                "total_positions": state["training_moves"] + measured["observations"],
                "prior_log_loss": measured["prior_log_loss"],
                "personal_log_loss": measured["personal_log_loss"],
                "prior_accuracy": measured["prior_top1"],
                "personal_accuracy": measured["personal_top1"], "trained_at": trained_at}

    try:
        while state["phase"] != "complete":
            phase = state["phase"]
            if phase == "evaluate":
                order = list(range(split, count))
            else:
                order = list(range(split if phase == "train" else count))
                random.Random(identity["shuffle_seed"] + state["epoch"] +
                              (10000 if phase == "deploy" else 0)).shuffle(order)
            for position in range(state["cursor"], len(order)):
                index = order[position]
                game = selected[index]
                try:
                    examples = game_examples(game, iterator)
                except ValueError as error:
                    if phase == "evaluate" or state["epoch"] == 0:
                        state["skipped"].append({"phase": phase, "url": game["url"], "reason": str(error)})
                    examples = []
                if phase == "evaluate":
                    for example in examples:
                        board, chosen = example["board"], example["move_uci"]
                        prior = prior_provider(board)
                        personal = policy.distribution(board, prior)
                        add_metrics(state["heldout_metrics"], prior, personal, chosen)
                        time_class = game.get("time_class") or "unknown"
                        add_metrics(state["by_time_class"].setdefault(time_class, empty_metrics()), prior, personal, chosen)
                    state["test_games_ok"] += bool(examples)
                else:
                    for start in range(0, len(examples), identity["batch_size"]):
                        batch = [{"board": example["board"], "chosen_uci": example["move_uci"],
                                  "prior": prior_provider(example["board"])}
                                 for example in examples[start:start + identity["batch_size"]]]
                        policy.fit_batch(batch, count_observations=state["epoch"] == 0, steps=1)
                    if state["epoch"] == 0:
                        state["training_moves" if phase == "train" else "deployment_moves"] += len(examples)
                        state["train_games_ok" if phase == "train" else "deploy_games_ok"] += bool(examples)
                state["cursor"] = position + 1
                state["last_completed_game"] = {"url": game["url"], "fingerprint": digest({
                    "pgn": game["pgn"], "user_color": game["user_color"]}),
                    "phase": phase, "epoch": state["epoch"]}
                save()
            if phase != "evaluate" and state["epoch"] + 1 < identity["fit_epochs"]:
                state["epoch"] += 1
                state["cursor"] = 0
                save()
                continue
            if phase == "train":
                recent_replay(range(split))
                snapshot("split_model.json", "frozen_training_80_percent")
                state["phase"], state["cursor"], state["epoch"] = "evaluate", 0, 0
            elif phase == "evaluate":
                metrics = summarize(state["heldout_metrics"])
                improvement = metrics["personal_log_loss_improvement"]
                report = {**identity, "evaluation": "frozen_chronological_game_holdout",
                          "summary": summary(), "training_complete": False,
                          "training_moves": state["training_moves"], "heldout": metrics,
                          "by_time_class": {name: summarize(value) for name, value in state["by_time_class"].items()},
                          "prediction_result": "no_heldout_observations" if improvement is None else
                          "personal_lower_log_loss" if improvement > 0 else "personal_not_lower_log_loss",
                          "skipped": state["skipped"], "source_manifest": source_manifest,
                          "limitation": "Observed prediction metrics do not establish opponent strength or improved human learning."}
                atomic_json(directory / "report.json", report)
                policy = PersonalPolicy()
                state["phase"], state["cursor"], state["epoch"] = "deploy", 0, 0
            else:
                from datetime import datetime, timezone
                recent_replay(range(count))
                final_summary = summary(datetime.now(timezone.utc).isoformat())
                if policy.samples != final_summary["total_positions"] or state["deploy_games_ok"] != final_summary["games_total"]:
                    raise ValueError("Deployment counts do not match the evaluated source partitions.")
                report = json.loads((directory / "report.json").read_text())
                report.update(summary=final_summary, training_complete=True)
                atomic_json(directory / "report.json", report)
                snapshot("deployment_model.json", "all_selected_games_for_live_play", final_summary)
                state["phase"] = "complete"
            save()
        progress.write("complete", force=True, games_done=count, total_games=count,
                       samples=policy.samples, report=str(directory / "report.json"),
                       deployment_model=str(directory / "deployment_model.json"))
        return json.loads((directory / "report.json").read_text())
    except BaseException as error:
        progress.write("failed", force=True, stage=state["phase"], error=str(error),
                       last_completed_game=state["last_completed_game"],
                       note="Resume from the last completed game; checkpoint was not advanced for a partial game.")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".runtime/players",
                        help="Archive-cache parent; importer creates a username subdirectory.")
    parser.add_argument("--output-dir", type=Path, help="Default: CACHE_DIR/USERNAME/training")
    parser.add_argument("--max-games", type=int, help="Smoke tests only; omitted means every eligible game.")
    parser.add_argument("--workers", type=int, default=4, help="Persistent parallel Maia workers during prior caching.")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--priors-only", action="store_true", help="Populate reusable priors, then stop before fitting.")
    stages.add_argument("--fit-only", action="store_true", help="Fit/resume using completed local archives and priors; no downloads or prior rescan.")
    args = parser.parse_args(argv)
    username = username_key(args.username)
    output = args.output_dir or args.cache_dir / username / "training"
    result = cached_games(username, args.cache_dir) if args.fit_only else fetch_games(username, args.cache_dir)
    games = selected_games(result.games, username, args.max_games)
    fingerprint = maia_fingerprint()
    with PriorCache(result.cache_dir / "priors.sqlite3", username, fingerprint) as cache:
        if not args.fit_only:
            precompute_priors(games, cache, output, username, workers=args.workers)
        if args.priors_only:
            return
        def cached_prior(board):
            prior = cache.get(board)
            if prior is None:
                raise RuntimeError("A required Maia prior was not precomputed; rerun to resume caching.")
            return prior
        run_training(result.games, username, output, cached_prior, policy_fingerprint=fingerprint,
                     max_games=args.max_games, source_manifest=result.manifest)


if __name__ == "__main__":
    main()
