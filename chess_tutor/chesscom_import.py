"""Read-only, resumable import of one requested Chess.com player's history.

Official API: https://www.chess.com/news/view/published-data-api
Only public archives are requested, serially and without authentication.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import chess
import chess.pgn

DEFAULT_USER_AGENT = "NEMESIS/0.2 (+https://github.com/TheOneWhoBurns/Chess-LLM-Tutor)"
_IMPORT_LOCK = threading.Lock()
_USERNAME = re.compile(r"[A-Za-z0-9_-]{1,50}\Z")
_MAX_RESPONSE_BYTES = 128 * 1024 * 1024


class ChessComImportError(RuntimeError):
    pass


class GameSkipped(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


@dataclass
class ImportResult:
    games: list[dict]
    manifest: dict
    cache_dir: Path


def _username(value):
    if not isinstance(value, str) or not _USERNAME.fullmatch(value):
        raise ValueError("Use the exact Chess.com username, containing only letters, digits, '_' or '-'.")
    return value.casefold()


class _QuietBuilder(chess.pgn.GameBuilder):
    def handle_error(self, error):
        self.game.errors.append(error)


def _parse_pgn(text):
    if not isinstance(text, str) or not text.strip():
        raise GameSkipped("missing_pgn")
    try:
        stream = io.StringIO(text)
        game = chess.pgn.read_game(stream, Visitor=_QuietBuilder)
        if game is None or game.errors or chess.pgn.read_game(stream, Visitor=_QuietBuilder) is not None:
            raise GameSkipped("invalid_pgn")
        board = game.board()
        if type(board) is not chess.Board or board.chess960:
            raise GameSkipped("nonstandard_rules")
        if not board.is_valid():
            raise GameSkipped("invalid_pgn")
        for move in game.mainline_moves():
            if move not in board.legal_moves:
                raise GameSkipped("invalid_pgn")
            board.push(move)
        return game
    except GameSkipped:
        raise
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise GameSkipped("invalid_pgn") from exc


def parse_game(raw: dict, username: str) -> dict:
    """Validate an archive record; API player names determine the user's color."""
    username = _username(username)
    if not isinstance(raw, dict):
        raise GameSkipped("invalid_record")
    if raw.get("rules") != "chess":
        raise GameSkipped("nonstandard_rules")
    colors = []
    for color in ("white", "black"):
        details = raw.get(color)
        name = details.get("username") if isinstance(details, dict) else None
        if isinstance(name, str) and name.casefold() == username:
            colors.append(color)
    if not colors:
        raise GameSkipped("username_mismatch")
    if len(colors) != 1:
        raise GameSkipped("ambiguous_user")
    color = colors[0]
    url = raw.get("url")
    if not isinstance(url, str) or not url:
        raise GameSkipped("missing_url")
    timestamp = raw.get("end_time")
    if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
        raise GameSkipped("invalid_timestamp")
    game = _parse_pgn(raw.get("pgn"))
    board = game.board()
    human_moves = 0
    for move in game.mainline_moves():
        human_moves += board.turn == (color == "white")
        board.push(move)
    if human_moves == 0:
        raise GameSkipped("no_human_moves")
    pgn = raw["pgn"]
    return {
        "url": url, "pgn": pgn, "user_color": color,
        "timestamp": timestamp, "end_time": timestamp,
        "start_time": raw.get("start_time"), "time_control": raw.get("time_control"),
        "time_class": raw.get("time_class"), "rated": raw.get("rated"),
        "result": raw[color].get("result"), "pgn_result": game.headers.get("Result"),
        "rules": "chess", "human_moves": human_moves,
        "white": raw.get("white"), "black": raw.get("black"),
        "pgn_sha256": hashlib.sha256(pgn.encode()).hexdigest(),
        "fingerprint": hashlib.sha256((url + "\n" + pgn + "\n" + color).encode()).hexdigest(),
    }


def iter_training_examples(record: dict):
    """Yield the player's observed choices with independent full-history boards.

    Split records by game before iterating, so positions from one game cannot
    leak across training and evaluation partitions.
    """
    color = record.get("user_color")
    if color not in ("white", "black"):
        raise ValueError("Training record must identify user_color as white or black.")
    game = _parse_pgn(record.get("pgn"))
    board = game.board()
    for ply, move in enumerate(game.mainline_moves(), 1):
        if board.turn == (color == "white"):
            yield {"board": board.copy(stack=True), "move_uci": move.uci(), "ply": ply,
                   "game_url": record["url"], "user_color": color}
        board.push(move)


def _atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, data):
    _atomic_bytes(path, json.dumps(data, indent=2, sort_keys=True).encode())


class _SamePlayerRedirect(HTTPRedirectHandler):
    def __init__(self, validate):
        self.validate = validate

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        self.validate(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


class ChessComImporter:
    def __init__(self, cache_dir, *, user_agent=DEFAULT_USER_AGENT, timeout=20,
                 max_retries=3, opener=None, sleep=None, now=None, progress=None):
        if not isinstance(user_agent, str) or not user_agent.strip() or "\n" in user_agent or "\r" in user_agent:
            raise ValueError("Provide a recognizable single-line User-Agent with contact information.")
        if not 0 < timeout <= 60 or not 0 <= max_retries <= 6:
            raise ValueError("Timeout must be in (0,60] seconds and max_retries in [0,6].")
        self.root = Path(cache_dir).expanduser().resolve()
        self.user_agent, self.timeout = user_agent, timeout
        self.max_retries, self.opener = max_retries, opener
        self.sleep = sleep or time.sleep
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.progress = progress

    def _validate_url(self, url):
        if not isinstance(url, str):
            raise ChessComImportError("Archive index contained a non-string URL.")
        parts = urlsplit(url)
        prefix = f"/pub/player/{self.username}/games/"
        if (parts.scheme != "https" or parts.netloc.casefold() != "api.chess.com"
                or parts.query or parts.fragment or not parts.path.casefold().startswith(prefix)):
            raise ChessComImportError("Archive URL does not belong to the requested player's public API.")
        suffix = parts.path[len(prefix):]
        if suffix != "archives" and not re.fullmatch(r"[0-9]{4}/(?:0[1-9]|1[0-2])", suffix):
            raise ChessComImportError("Archive index contained an invalid year/month URL.")
        return suffix

    def _request_json(self, url):
        self._validate_url(url)
        opener = self.opener or build_opener(_SamePlayerRedirect(self._validate_url)).open
        for attempt in range(self.max_retries + 1):
            request = Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
            try:
                with opener(request, timeout=self.timeout) as response:
                    self._validate_url(response.geturl())
                    payload = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(payload) > _MAX_RESPONSE_BYTES:
                    raise ChessComImportError("Archive exceeded the supported 128 MiB response limit.")
                try:
                    data = json.loads(payload)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ChessComImportError(f"Chess.com returned invalid JSON for {url}.") from exc
                if not isinstance(data, dict):
                    raise ChessComImportError("Chess.com returned an unexpected JSON structure.")
                return data, payload
            except HTTPError as exc:
                status = exc.code
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                exc.close()
                if status != 429 and not 500 <= status <= 599:
                    raise ChessComImportError(f"Chess.com returned HTTP {status} for {url}; no archive was overwritten.") from exc
                error = f"HTTP {status}"
            except (URLError, TimeoutError, ConnectionError) as exc:
                retry_after = None
                error = str(exc)
            if attempt >= self.max_retries:
                raise ChessComImportError(f"Chess.com download failed after {attempt + 1} attempts: {error}.")
            delay = min(2 ** attempt, 30)
            if retry_after:
                try:
                    requested = float(retry_after)
                except ValueError:
                    try:
                        requested = (parsedate_to_datetime(retry_after) - self.now()).total_seconds()
                    except (ValueError, TypeError, OverflowError):
                        requested = 0
                if requested > 30:
                    raise ChessComImportError(f"Chess.com asked to retry after {requested:g} seconds; retry the resumable import later.")
                delay = max(delay, requested)
            self.sleep(delay)

    def fetch_games(self, username):
        with _IMPORT_LOCK:
            return self._fetch_games(username)

    def _fetch_games(self, username):
        self.username = _username(username)
        directory = self.root / self.username
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": 1, "username": self.username,
                    "source": "https://www.chess.com/news/view/published-data-api",
                    "fetched_at": self.now().isoformat(), "status": "downloading",
                    "archive_count": 0, "archives_completed": 0,
                    "raw_games": 0, "unique_games": 0, "accepted_games": 0,
                    "human_moves": 0, "skipped": {}, "archives": []}
        games, seen, skipped = [], set(), Counter()
        manifest_path = directory / "manifest.json"
        try:
            index_url = f"https://api.chess.com/pub/player/{self.username}/games/archives"
            index, raw_index = self._request_json(index_url)
            urls = index.get("archives")
            if not isinstance(urls, list):
                raise ChessComImportError("Chess.com archive index is missing its archives list.")
            months = {}
            for url in urls:
                month = self._validate_url(url)
                if month == "archives":
                    raise ChessComImportError("Archive list contained its own index URL.")
                months[month] = url
            _atomic_bytes(directory / "archives.json", raw_index)
            manifest["archive_count"] = len(months)
            _write_json(manifest_path, manifest)
            current_month = self.now().strftime("%Y/%m")
            for month, url in sorted(months.items()):
                path = directory / (month.replace("/", "-") + ".json")
                cached = False
                data = None
                if month < current_month and path.is_file():
                    try:
                        payload = path.read_bytes()
                        data = json.loads(payload)
                        if not isinstance(data, dict) or not isinstance(data.get("games"), list):
                            data = None
                        else:
                            cached = True
                    except (OSError, ValueError):
                        data = None
                if data is None:
                    data, payload = self._request_json(url)
                    if not isinstance(data.get("games"), list):
                        raise ChessComImportError(f"Archive {month} is missing its games list.")
                    _atomic_bytes(path, payload)
                for raw in data["games"]:
                    manifest["raw_games"] += 1
                    game_url = raw.get("url") if isinstance(raw, dict) else None
                    if isinstance(game_url, str) and game_url:
                        if game_url in seen:
                            skipped["duplicate_url"] += 1
                            continue
                        seen.add(game_url)
                    try:
                        game = parse_game(raw, self.username)
                    except GameSkipped as exc:
                        skipped[exc.reason] += 1
                        continue
                    games.append(game)
                    manifest["human_moves"] += game["human_moves"]
                manifest.update(archives_completed=manifest["archives_completed"] + 1,
                                unique_games=len(seen), accepted_games=len(games), skipped=dict(skipped))
                manifest["archives"].append({"month": month, "url": url, "file": path.name,
                                              "sha256": hashlib.sha256(payload).hexdigest(),
                                              "game_count": len(data["games"]), "cached": cached})
                _write_json(manifest_path, manifest)
                if self.progress:
                    self.progress(dict(manifest))
            games.sort(key=lambda game: (game["timestamp"], game["url"]))
            manifest.update(status="complete", first_game_time=games[0]["timestamp"] if games else None,
                            last_game_time=games[-1]["timestamp"] if games else None)
            _write_json(manifest_path, manifest)
            return ImportResult(games=games, manifest=manifest, cache_dir=directory)
        except Exception as exc:
            manifest.update(status="failed", error=str(exc), skipped=dict(skipped))
            _write_json(manifest_path, manifest)
            raise


def fetch_games(username, cache_dir, **options):
    return ChessComImporter(cache_dir, **options).fetch_games(username)
