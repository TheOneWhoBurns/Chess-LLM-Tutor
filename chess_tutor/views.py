"""Single-player API. Cookies identify the personal profile; no account required."""
import json
import logging

import chess
import chess.pgn
from django.db import OperationalError, transaction
from django.db.models import F
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from .models import PlayerProfile
from .nemesis import Evaluator, PlayerNetwork, choose_move, features, observe, weakness_summary

LOGGER = logging.getLogger(__name__)


def initial_state():
    return {"network": PlayerNetwork().dump(), "stats": {}, "moves": [], "events": [],
            "games": 1, "completed": 0, "total_loss": 0, "mode": "adaptive", "archive": [],
            "engine": "Demo search · 2 ply", "result": "*"}


def profile_for(request):
    profile_id = request.session.get("nemesis_profile")
    profile = PlayerProfile.objects.filter(pk=profile_id).first() if profile_id else None
    if profile is None:
        profile = PlayerProfile.objects.create(state=initial_state())
        request.session["nemesis_profile"] = str(profile.pk)
    return profile


def restore_board(state):
    board = chess.Board()
    for uci in state["moves"]:
        board.push_uci(uci)
    return board


def public_state(profile):
    state = profile.state
    board = chess.Board()
    history = []
    for uci in state["moves"]:
        move = board.parse_uci(uci)
        history.append(board.san(move))
        board.push(move)
    net = PlayerNetwork(state["network"])
    rows = weakness_summary(state["stats"])
    supported = [r for r in rows if r["supported"] and r["errors"]]
    target = max(supported, key=lambda r: r["mean_loss"])["name"] if supported else None
    game_over = state["result"] != "*" or board.is_game_over()
    return {"revision": profile.revision, "fen": board.fen(), "moves": history,
            "legal_moves": [m.uci() for m in board.legal_moves] if not game_over else [],
            "last_move": state["moves"][-1] if state["moves"] else None,
            "in_check": board.is_check(), "game_over": game_over, "result": state["result"],
            "turn": "white" if board.turn else "black", "mode": state["mode"],
            "events": state["events"][-30:], "archive": state["archive"],
            "profile": {"samples": net.samples, "phase": "Adapting" if net.influence else "Calibrating",
                        "influence": round(net.influence * 100), "weaknesses": rows,
                        "target": target, "games": state["games"], "completed": state["completed"],
                        "mean_loss": round(state["total_loss"] / net.samples) if net.samples else None},
            "engine": state["engine"]}


@ensure_csrf_cookie
@require_GET
def chat_view(request):
    profile_for(request)
    return render(request, "chat.html")


@require_GET
def state_view(request):
    return JsonResponse(public_state(profile_for(request)))


def apply_move(state, text):
    board = restore_board(state)
    if state["result"] != "*" or board.is_game_over():
        raise ValueError("This game has ended. Start a new game to play again.")
    if board.turn != chess.WHITE:
        raise ValueError("Wait for NEMESIS to finish its turn.")
    clean = text.strip().replace("0-0", "O-O")
    try:
        move = board.parse_san(clean)
    except ValueError:
        move = board.parse_uci(clean.replace("-", ""))
    if move not in board.legal_moves:
        raise ValueError("That move is not legal in this position.")
    x = features(board)
    network = PlayerNetwork(state["network"])
    san = board.san(move)
    move_number = board.fullmove_number
    with Evaluator() as evaluator:
        ranked = evaluator.rank(board)
        chosen_score = next(c["score"] for c in ranked if c["move"] == move)
        loss = max(0., ranked[0]["score"] - chosen_score)
        best_san = board.san(ranked[0]["move"])
        network.learn(x, loss)
        observe(state["stats"], x, loss)
        board.push(move)
        state["moves"].append(move.uci())
        event = {"number": move_number, "player": san, "loss_cp": round(loss),
                 "alternative": best_san, "adapted": False, "mode": state["mode"],
                 "training_engine": evaluator.name,
                 "quality": "Blunder" if loss >= 200 else "Mistake" if loss >= 100 else "Steady"}
        if not board.is_game_over():
            candidates = evaluator.rank(board)
            selected, changed = choose_move(board, candidates, network, state["mode"] == "adaptive")
            reply = selected["move"]
            event["opponent"] = board.san(reply)
            event["adapted"] = changed
            event["search_cost_cp"] = round(candidates[0]["score"] - selected["score"])
            board.push(reply)
            state["moves"].append(reply.uci())
        state["engine"] = evaluator.name
        event["engine"] = evaluator.name
    state["network"] = network.dump()
    state["total_loss"] += round(min(loss, 1000))
    state["events"].append(event)
    if board.is_game_over():
        state["result"] = board.result()
        state["completed"] += 1


def new_game(state, mode):
    if mode not in ("adaptive", "baseline"):
        raise ValueError("Choose adaptive or baseline mode.")
    if state["moves"]:
        state["archive"] = ([{"game": state["games"], "moves": list(state["moves"]),
                              "events": list(state["events"]), "mode": state["mode"],
                              "result": state["result"]}] + state["archive"])[:30]
    state.update(moves=[], events=[], result="*", mode=mode, games=state["games"] + 1)


@require_POST
def action_view(request):
    try:
        data = json.loads(request.body)
        if not isinstance(data, dict):
            raise ValueError("Send a JSON object.")
        if type(data.get("revision")) is not int:
            raise ValueError("A board revision is required. Refresh and try again.")
        action = data.get("action", "move")
        if action not in ("move", "new_game", "forget", "resign", "claim_draw"):
            raise ValueError("Unknown action.")
        profile = profile_for(request)
        with transaction.atomic():
            # Acquire a write lock before reading state, also on SQLite. Revisions
            # prevent a retried request or stale tab from applying a move twice.
            updated = PlayerProfile.objects.filter(pk=profile.pk, revision=data["revision"]).update(
                revision=F("revision") + 1)
            if not updated:
                profile.refresh_from_db()
                return JsonResponse({"error": "The board changed. Your position has been refreshed.",
                                     "state": public_state(profile)}, status=409)
            profile.refresh_from_db()
            state = profile.state
            if action == "move":
                move = data.get("move", data.get("message", ""))
                if not isinstance(move, str) or not 1 <= len(move) <= 16:
                    raise ValueError("Enter a move such as e4, Nf3, O-O, or e2e4.")
                apply_move(state, move)
            elif action == "new_game":
                new_game(state, data.get("mode", state["mode"]))
            elif action == "forget":
                state = initial_state()
            else:
                board = restore_board(state)
                if state["result"] != "*" or board.is_game_over():
                    raise ValueError("This game has already ended.")
                if action == "claim_draw" and not board.can_claim_draw():
                    raise ValueError("A draw cannot be claimed in this position.")
                state["result"] = "0-1" if action == "resign" else "1/2-1/2"
                state["completed"] += 1
            profile.state = state
            profile.save(update_fields=["state", "updated_at"])
            result = public_state(profile)
        return JsonResponse(result)
    except (ValueError, UnicodeDecodeError) as error:
        return JsonResponse({"error": str(error)}, status=400)
    except OperationalError:
        return JsonResponse({"error": "NEMESIS is busy. Please retry your move."}, status=409)
    except Exception:
        LOGGER.exception("NEMESIS action failed; transaction rolled back")
        return JsonResponse({"error": "The move could not be completed. Your saved position is unchanged."}, status=500)


@require_GET
def export_view(request):
    profile = profile_for(request)
    if request.GET.get("format") == "json":
        response = JsonResponse({"version": 1, "state": profile.state}, json_dumps_params={"indent": 2})
        response["Content-Disposition"] = 'attachment; filename="nemesis-research.json"'
        return response
    game = chess.pgn.Game.from_board(restore_board(profile.state))
    game.headers.update(Event="NEMESIS personal training", White="You", Black="NEMESIS",
                        Result=profile.state["result"], Mode=profile.state["mode"])
    response = HttpResponse(str(game), content_type="application/x-chess-pgn")
    response["Content-Disposition"] = 'attachment; filename="nemesis-game.pgn"'
    return response
