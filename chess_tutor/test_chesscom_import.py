"""Public archive import contracts, exercised without network access."""
import copy
import hashlib
import json
import tempfile
import unittest
from collections import defaultdict, deque
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import chess

from .chesscom_import import (
    ChessComImportError,
    ChessComImporter,
    GameSkipped,
    fetch_games,
    iter_training_examples,
    parse_game,
)


NOW = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
INDEX = "https://api.chess.com/pub/player/owner/games/archives"
JULY = "https://api.chess.com/pub/player/owner/games/2026/07"
AUGUST = "https://api.chess.com/pub/player/owner/games/2026/08"
SEPTEMBER = "https://api.chess.com/pub/player/owner/games/2026/09"
PGN = '''[Event "Live Chess"]
[White "Owner"]
[Black "Opponent"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *
'''


def game(number=1, *, owner_color="white", pgn=PGN, **changes):
    raw = {
        "url": f"https://www.chess.com/game/live/{number}",
        "pgn": pgn,
        "end_time": int(NOW.timestamp()) - 1000 + number,
        "time_control": "600",
        "time_class": "rapid",
        "rated": True,
        "rules": "chess",
        "white": {"username": "Owner", "result": "win"},
        "black": {"username": "Opponent", "result": "resigned"},
    }
    if owner_color == "black":
        raw["white"]["username"] = "Opponent"
        raw["black"]["username"] = "Owner"
    raw.update(changes)
    return raw


class FakeResponse:
    def __init__(self, url, payload, headers=None):
        self.url = url
        self.body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.headers = headers or {}
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size=-1):
        return self.body if size < 0 else self.body[:size]

    def geturl(self):
        return self.url


class ScriptedOpener:
    """Every request consumes a response; unexpected requests fail immediately."""

    def __init__(self):
        self.responses = defaultdict(deque)
        self.calls = []

    def add(self, url, *responses):
        self.responses[url].extend(responses)
        return self

    def __call__(self, request, *, timeout):
        url = request.full_url
        self.calls.append((request, timeout))
        if not self.responses[url]:
            raise AssertionError(f"Unexpected network request: {url}")
        response = self.responses[url].popleft()
        if isinstance(response, Exception):
            raise response
        if isinstance(response, FakeResponse):
            return response
        return FakeResponse(url, response)

    @property
    def urls(self):
        return [request.full_url for request, _ in self.calls]


def http_error(url, code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return HTTPError(url, code, "scripted public API failure", headers, None)


class GameParsingTests(unittest.TestCase):
    def test_api_identity_controls_color_with_exact_case_insensitive_matching(self):
        for color, expected in (
            ("white", ["e2e4", "g1f3", "f1b5"]),
            ("black", ["e7e5", "b8c6", "a7a6"]),
        ):
            with self.subTest(color=color):
                # PGN labels can disagree; authoritative API player fields win.
                raw = game(owner_color=color, pgn=PGN.replace('"Owner"', '"Different"'))
                original = copy.deepcopy(raw)
                record = parse_game(raw, "oWnEr")
                examples = list(iter_training_examples(record))
                self.assertEqual(record["user_color"], color)
                self.assertEqual(record["human_moves"], 3)
                self.assertEqual([item["move_uci"] for item in examples], expected)
                self.assertEqual([item["ply"] for item in examples],
                                 [1, 3, 5] if color == "white" else [2, 4, 6])
                self.assertTrue(all(item["user_color"] == color for item in examples))
                self.assertTrue(all(item["game_url"] == raw["url"] for item in examples))
                self.assertEqual(record["pgn_sha256"], hashlib.sha256(raw["pgn"].encode()).hexdigest())
                self.assertTrue(record["fingerprint"])
                self.assertEqual(record["rules"], "chess")
                self.assertEqual(raw, original)

    def test_examples_preserve_history_and_each_board_is_independent(self):
        pgn = PGN.split("\n\n", 1)[0] + "\n\n1. Nf3 Nf6 2. Ng1 Ng8 3. e4 e5 *\n"
        record = parse_game(game(pgn=pgn), "owner")
        examples = list(iter_training_examples(record))
        third = examples[2]["board"]
        self.assertEqual(len(third.move_stack), 4)
        self.assertTrue(third.is_repetition(2))
        self.assertEqual(examples[0]["board"].fen(), chess.STARTING_FEN)
        for item in examples:
            self.assertIn(chess.Move.from_uci(item["move_uci"]), item["board"].legal_moves)
            self.assertEqual(len(item["board"].move_stack), item["ply"] - 1)
        third.push_uci("e2e4")
        self.assertEqual(len(examples[0]["board"].move_stack), 0)
        self.assertEqual(len(list(iter_training_examples(record))[2]["board"].move_stack), 4)

    def test_underpromotion_is_preserved_as_a_legal_choice(self):
        pgn = '''[White "Owner"]
[Black "Opponent"]
[SetUp "1"]
[FEN "7k/P7/8/8/8/8/8/7K w - - 0 1"]
[Result "*"]

1. a8=N *
'''
        record = parse_game(game(pgn=pgn), "owner")
        examples = list(iter_training_examples(record))
        self.assertEqual(record["human_moves"], 1)
        self.assertEqual(examples[0]["move_uci"], "a7a8n")
        self.assertIn(chess.Move.from_uci("a7a8n"), examples[0]["board"].legal_moves)

    def test_bad_games_have_explicit_skip_reasons(self):
        mismatch = game()
        mismatch["white"]["username"] = "owner_extra"
        ambiguous = game()
        ambiguous["black"]["username"] = "OWNER"
        cases = [
            (game(pgn=""), "missing_pgn"),
            (game(pgn='[Result "*"]\n\n1. e4 e5 2. e5 *'), "invalid_pgn"),
            (game(rules="chess960"), "nonstandard_rules"),
            (mismatch, "username_mismatch"),
            (ambiguous, "ambiguous_user"),
            (game(url=""), "missing_url"),
            (game(end_time="not-a-timestamp"), "invalid_timestamp"),
            (game(pgn='[Result "*"]\n\n*'), "no_human_moves"),
        ]
        for raw, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaises(GameSkipped) as caught:
                    parse_game(raw, "owner")
                self.assertEqual(caught.exception.reason, reason)

    def test_malformed_pgn_is_not_accepted_as_its_legal_prefix(self):
        raw = game(pgn='[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 3. e5 *')
        with self.assertRaises(GameSkipped) as caught:
            parse_game(raw, "owner")
        self.assertEqual(caught.exception.reason, "invalid_pgn")


class ChessComImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name).resolve()
        self.opener = ScriptedOpener()
        self.sleeps = []

    def importer(self, **options):
        return ChessComImporter(
            self.cache, opener=self.opener, sleep=self.sleeps.append,
            now=lambda: NOW, **options,
        )

    def test_downloads_deduplicates_sorts_and_writes_reviewable_cache(self):
        older = game(1, end_time=int(NOW.timestamp()) - 2000)
        newer = game(2, owner_color="black")
        self.opener.add(INDEX, {"archives": [AUGUST, JULY]})
        self.opener.add(JULY, {"games": [newer, older]})
        self.opener.add(AUGUST, {"games": [copy.deepcopy(older)]})
        result = self.importer(user_agent="NEMESIS test contact", timeout=7).fetch_games("OwNeR")
        self.assertEqual([item["url"] for item in result.games], [older["url"], newer["url"]])
        self.assertEqual(result.cache_dir, self.cache / "owner")
        for key, expected in {
            "archive_count": 2, "archives_completed": 2, "raw_games": 3,
            "unique_games": 2, "accepted_games": 2, "human_moves": 6,
        }.items():
            self.assertEqual(result.manifest[key], expected, key)
        self.assertEqual(json.loads((result.cache_dir / "archives.json").read_text()),
                         {"archives": [AUGUST, JULY]})
        self.assertEqual(json.loads((result.cache_dir / "2026-07.json").read_text()),
                         {"games": [newer, older]})
        saved_manifest = json.loads((result.cache_dir / "manifest.json").read_text())
        self.assertEqual(saved_manifest["accepted_games"], 2)
        self.assertTrue(all(timeout == 7 for _, timeout in self.opener.calls))
        self.assertTrue(all(request.get_header("User-agent") == "NEMESIS test contact"
                            for request, _ in self.opener.calls))

    def test_variants_and_invalid_games_are_counted_without_partial_training_moves(self):
        invalid = game(3, pgn='[Result "*"]\n\n1. e4 e5 2. e5 *')
        self.opener.add(INDEX, {"archives": [AUGUST]})
        self.opener.add(AUGUST, {"games": [game(1), game(2, rules="bughouse"), invalid]})
        result = self.importer().fetch_games("owner")
        self.assertEqual(result.manifest["raw_games"], 3)
        self.assertEqual(result.manifest["accepted_games"], 1)
        self.assertEqual(result.manifest["human_moves"], 3)
        self.assertEqual(result.manifest["skipped"]["nonstandard_rules"], 1)
        self.assertEqual(result.manifest["skipped"]["invalid_pgn"], 1)
        self.assertEqual(sum(len(list(iter_training_examples(item))) for item in result.games), 3)

    def test_repeat_fetch_reuses_closed_month_but_refreshes_index_and_current_month(self):
        self.opener.add(INDEX, {"archives": [AUGUST, SEPTEMBER]}, {"archives": [AUGUST, SEPTEMBER]})
        self.opener.add(AUGUST, {"games": [game(1)]})
        self.opener.add(SEPTEMBER, {"games": [game(2)]}, {"games": [game(2), game(3)]})
        first = self.importer().fetch_games("owner")
        second = self.importer().fetch_games("OWNER")
        self.assertEqual(first.manifest["accepted_games"], 2)
        self.assertEqual(second.manifest["accepted_games"], 3)
        self.assertEqual(self.opener.urls.count(INDEX), 2)
        self.assertEqual(self.opener.urls.count(AUGUST), 1)
        self.assertEqual(self.opener.urls.count(SEPTEMBER), 2)
        self.assertEqual(json.loads((second.cache_dir / "2026-09.json").read_text())["games"],
                         [game(2), game(3)])

    def test_corrupt_closed_month_is_refetched(self):
        cache_dir = self.cache / "owner"
        cache_dir.mkdir()
        (cache_dir / "2026-08.json").write_text("{broken json")
        self.opener.add(INDEX, {"archives": [AUGUST]})
        self.opener.add(AUGUST, {"games": [game()]})
        result = self.importer().fetch_games("owner")
        self.assertEqual(result.manifest["accepted_games"], 1)
        self.assertIn(AUGUST, self.opener.urls)
        self.assertEqual(json.loads((cache_dir / "2026-08.json").read_text()), {"games": [game()]})

    def test_invalid_usernames_are_rejected_before_network_or_cache_creation(self):
        for username in ("", "../owner", "owner/name", "owner name", " owner", "owner\n",
                         "ownér", "a" * 51, None):
            with self.subTest(username=username):
                with self.assertRaises(ValueError):
                    self.importer().fetch_games(username)
        self.assertEqual(self.opener.urls, [])
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_untrusted_archive_urls_are_rejected_before_fetching_them(self):
        urls = [
            "https://example.com/pub/player/owner/games/2026/08",
            "http://api.chess.com/pub/player/owner/games/2026/08",
            "https://api.chess.com/pub/player/another/games/2026/08",
            "https://api.chess.com/pub/player/owner/games/2026/13",
            "https://api.chess.com/pub/player/owner/games/2026/08?extra=1",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.opener.add(INDEX, {"archives": [url]})
                with self.assertRaises(ChessComImportError):
                    self.importer().fetch_games("owner")
        self.assertEqual(self.opener.urls, [INDEX] * len(urls))

    def test_rate_limit_and_transient_server_error_retry_then_succeed(self):
        self.opener.add(INDEX, http_error(INDEX, 429, 2), http_error(INDEX, 503), {"archives": []})
        result = self.importer(max_retries=2).fetch_games("owner")
        self.assertEqual(result.games, [])
        self.assertEqual(self.opener.urls, [INDEX] * 3)
        self.assertEqual(len(self.sleeps), 2)
        self.assertGreaterEqual(self.sleeps[0], 2)
        self.assertTrue(all(delay > 0 for delay in self.sleeps))

    def test_retries_are_bounded_and_not_found_is_not_retried(self):
        for code, attempts in ((503, 3), (404, 1)):
            with self.subTest(code=code):
                start = len(self.opener.calls)
                sleep_start = len(self.sleeps)
                self.opener.add(INDEX, *(http_error(INDEX, code) for _ in range(attempts)))
                with self.assertRaises(ChessComImportError):
                    self.importer(max_retries=2).fetch_games("owner")
                self.assertEqual(len(self.opener.calls) - start, attempts)
                self.assertEqual(len(self.sleeps) - sleep_start, attempts - 1)

    def test_long_retry_after_stops_without_retrying_before_allowed_time(self):
        self.opener.add(INDEX, http_error(INDEX, 429, 120))
        with self.assertRaises(ChessComImportError):
            self.importer(max_retries=2).fetch_games("owner")
        self.assertEqual(self.opener.urls, [INDEX])
        self.assertEqual(self.sleeps, [])

    def test_partial_manifest_survives_failure_and_completed_month_is_reused(self):
        self.opener.add(INDEX, {"archives": [JULY, AUGUST]}, {"archives": [JULY, AUGUST]})
        self.opener.add(JULY, {"games": [game(1)]})
        self.opener.add(AUGUST, http_error(AUGUST, 503), {"games": [game(2)]})
        with self.assertRaises(ChessComImportError):
            self.importer(max_retries=0).fetch_games("owner")
        manifest = json.loads((self.cache / "owner" / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["archives_completed"], 1)
        self.assertEqual(manifest["accepted_games"], 1)
        result = self.importer(max_retries=0).fetch_games("owner")
        self.assertEqual(result.manifest["accepted_games"], 2)
        self.assertEqual(self.opener.urls.count(JULY), 1)

    def test_top_level_helper_passes_injected_dependencies(self):
        self.opener.add(INDEX, {"archives": []})
        result = fetch_games("owner", self.cache, opener=self.opener,
                             sleep=self.sleeps.append, now=lambda: NOW)
        self.assertEqual(result.games, [])
        self.assertEqual(result.manifest["archive_count"], 0)
        self.assertEqual(result.manifest["human_moves"], 0)
        self.assertTrue((result.cache_dir / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
