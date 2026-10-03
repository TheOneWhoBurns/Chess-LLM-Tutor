"""History is a read-only view of the same replay that supplies the current board."""
from copy import deepcopy
from unittest.mock import patch

import chess
from django.test import TestCase

from .models import PlayerProfile
from .views import initial_state


class BoardHistoryTests(TestCase):
    def setUp(self):
        room = patch("chess_tutor.views.engine_room")
        self.room = room.start().return_value
        self.room.status.return_value = {"ready": True}
        self.addCleanup(room.stop)
        active = patch("chess_tutor.views.active_username", return_value=None)
        active.start()
        self.addCleanup(active.stop)
        self.client.get("/")
        self.profile = PlayerProfile.objects.get(pk=self.client.session["nemesis_profile"])

    def save_game(self, sans):
        board = chess.Board()
        moves = []
        expected = [{"fen": board.fen(), "last_move": None, "in_check": False}]
        for san in sans:
            move = board.parse_san(san)
            moves.append(move.uci())
            board.push(move)
            expected.append({"fen": board.fen(), "last_move": move.uci(), "in_check": board.is_check()})
        self.profile.state = initial_state()
        self.profile.state["moves"] = moves
        self.profile.revision = 7
        self.profile.save()
        return expected

    def assert_history(self, sans):
        expected = self.save_game(sans)
        saved_state = deepcopy(self.profile.state)
        saved_revision = self.profile.revision
        saved_updated_at = self.profile.updated_at
        response = self.client.get("/api/state/")
        self.assertEqual(response.status_code, 200)
        public = response.json()
        self.assertEqual(public["timeline"], expected)
        self.assertEqual(len(public["timeline"]), len(public["moves"]) + 1)
        self.assertEqual(public["moves"], sans)
        for field in ("fen", "last_move", "in_check"):
            self.assertEqual(public[field], public["timeline"][-1][field])
        live_board = chess.Board(public["fen"])
        self.assertEqual(list(public["legal_positions"]), public["legal_moves"])
        self.assertEqual(set(public["legal_moves"]),
                         {move.uci() for move in live_board.legal_moves} if not public["game_over"] else set())
        for uci, preview in public["legal_positions"].items():
            expected_board = live_board.copy()
            move = expected_board.parse_uci(uci)
            expected_san = expected_board.san(move)
            expected_board.push(move)
            self.assertEqual(preview, {"fen": expected_board.fen(), "last_move": uci,
                                       "in_check": expected_board.is_check(), "san": expected_san})
        # Loading or navigating a history must not become a training observation,
        # a rewind of the live game, or an additional revision.
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.state, saved_state)
        self.assertEqual(self.profile.revision, saved_revision)
        self.assertEqual(self.profile.updated_at, saved_updated_at)
        self.assertEqual(self.client.get("/api/state/").json(), public)
        return public

    def test_initial_position_and_completed_game(self):
        start = self.assert_history([])
        self.assertEqual(start["timeline"], [{"fen": chess.STARTING_FEN, "last_move": None, "in_check": False}])
        finished = self.assert_history(["f3", "e5", "g4", "Qh4#"])
        self.assertTrue(finished["timeline"][-1]["in_check"])
        self.assertTrue(finished["game_over"])
        self.assertEqual(finished["legal_moves"], [])
        self.assertEqual(finished["legal_positions"], {})
        self.assertFalse(finished["timeline"][-2]["in_check"])

    def test_castling_preserves_both_piece_positions_and_rights(self):
        public = self.assert_history(["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O"])
        before = chess.Board(public["timeline"][-2]["fen"])
        after = chess.Board(public["timeline"][-1]["fen"])
        self.assertEqual(before.king(chess.WHITE), chess.E1)
        self.assertEqual(after.king(chess.WHITE), chess.G1)
        self.assertEqual(after.piece_at(chess.F1), chess.Piece(chess.ROOK, chess.WHITE))
        self.assertIsNone(after.piece_at(chess.H1))
        self.assertTrue(before.has_kingside_castling_rights(chess.WHITE))
        self.assertFalse(after.has_castling_rights(chess.WHITE))

    def test_en_passant_replay_removes_the_captured_pawn(self):
        public = self.assert_history(["e4", "a6", "e5", "d5", "exd6"])
        before = chess.Board(public["timeline"][-2]["fen"])
        after = chess.Board(public["timeline"][-1]["fen"])
        self.assertEqual(before.ep_square, chess.D6)
        self.assertEqual(before.piece_at(chess.D5), chess.Piece(chess.PAWN, chess.BLACK))
        self.assertIsNone(after.piece_at(chess.D5))
        self.assertEqual(after.piece_at(chess.D6), chess.Piece(chess.PAWN, chess.WHITE))

    def test_underpromotion_replay_uses_the_chosen_piece(self):
        public = self.assert_history(["a4", "h5", "a5", "h4", "a6", "h3", "axb7", "hxg2", "bxa8=N"])
        after = chess.Board(public["timeline"][-1]["fen"])
        self.assertEqual(public["last_move"], "b7a8n")
        self.assertEqual(after.piece_at(chess.A8), chess.Piece(chess.KNIGHT, chess.WHITE))
        self.assertIsNone(after.piece_at(chess.B7))

    def test_special_move_previews_are_ready_before_the_move_is_played(self):
        castle = self.assert_history(["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"])
        preview = castle["legal_positions"]["e1g1"]
        castled = chess.Board(preview["fen"])
        self.assertEqual(preview["san"], "O-O")
        self.assertEqual(castled.king(chess.WHITE), chess.G1)
        self.assertEqual(castled.piece_at(chess.F1), chess.Piece(chess.ROOK, chess.WHITE))
        self.assertIsNone(castled.piece_at(chess.H1))
        self.assertEqual(chess.Board(castle["fen"]).king(chess.WHITE), chess.E1)

        en_passant = self.assert_history(["e4", "a6", "e5", "d5"])
        preview = en_passant["legal_positions"]["e5d6"]
        captured = chess.Board(preview["fen"])
        self.assertEqual(preview["san"], "exd6")
        self.assertIsNone(captured.piece_at(chess.D5))
        self.assertEqual(captured.piece_at(chess.D6), chess.Piece(chess.PAWN, chess.WHITE))
        self.assertEqual(chess.Board(en_passant["fen"]).piece_at(chess.D5), chess.Piece(chess.PAWN, chess.BLACK))

        promotion = self.assert_history(["a4", "h5", "a5", "h4", "a6", "h3", "axb7", "hxg2"])
        for symbol in "qrbn":
            preview = promotion["legal_positions"][f"b7a8{symbol}"]
            promoted = chess.Board(preview["fen"])
            self.assertEqual(promoted.piece_at(chess.A8), chess.Piece.from_symbol(symbol.upper()))
            self.assertIsNone(promoted.piece_at(chess.B7))
        self.assertEqual(chess.Board(promotion["fen"]).piece_at(chess.B7), chess.Piece(chess.PAWN, chess.WHITE))

    def test_resignation_has_no_previews_despite_legal_board_moves(self):
        self.save_game(["e4", "e5"])
        self.profile.state["result"] = "0-1"
        self.profile.save()
        saved_state = deepcopy(self.profile.state)
        public = self.client.get("/api/state/").json()
        self.assertTrue(chess.Board(public["fen"]).legal_moves.count())
        self.assertEqual(public["legal_positions"], {})
        self.assertEqual(public["legal_moves"], [])
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.state, saved_state)
