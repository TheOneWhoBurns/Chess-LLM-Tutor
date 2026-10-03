"""Single-player API with a local trained profile or a browser session profile."""
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
from .personal_profile import active_username, named_profile_id
from .player_policy import PersonalPolicy
from .runtime_identity import current_maia_fingerprint
from .opponent import EngineUnavailable, MaiaUnavailable, engine_room

LOGGER = logging.getLogger(__name__)


def initial_state():
    return {"version": 2, "policy": PersonalPolicy().dump(), "moves": [], "events": [],
            "games": 1, "completed": 0, "total_loss": 0, "mode": "adaptive", "archive": [],
            "decision": None, "result": "*", "metrics": {"count": 0, "prior_loss": 0., "personal_loss": 0.,
            "prior_hits": 0, "personal_hits": 0}}


def profile_for(request):
    profile_id = request.session.get("nemesis_profile")
    # This local installation belongs to one person. A trained Chess.com model
    # can become the default across this person's browsers without an account UI.
    configured = active_username()
    if configured:
        named_id = str(named_profile_id(configured))
        if PlayerProfile.objects.filter(pk=named_id).exists():
            profile_id = named_id
            if request.session.get("nemesis_profile") != named_id:
                request.session["nemesis_profile"] = named_id
    profile = PlayerProfile.objects.filter(pk=profile_id).first() if profile_id else None
    if profile is None:
        profile = PlayerProfile.objects.create(state=initial_state())
        request.session["nemesis_profile"] = str(profile.pk)
    elif profile.state.get("version") != 2:
        # Regret-regressor weights cannot be interpreted as a move-choice policy.
        # Preserve old research data and the active board; start new policy metrics.
        old = profile.state
        migrated = initial_state()
        migrated.update({key: old[key] for key in ("moves", "games", "completed", "mode", "archive", "result") if key in old})
        migrated["legacy_v1"] = old
        PlayerProfile.objects.filter(pk=profile.pk, revision=profile.revision).update(
            state=migrated, revision=F("revision") + 1)
        profile.refresh_from_db()
    return profile


def restore_board(state):
    board = chess.Board()
    for uci in state["moves"]:
        board.push_uci(uci)
    return board


def runtime_error(state):
    expected = state.get("policy_fingerprint")
    if expected:
        try:
            if expected == current_maia_fingerprint():
                return None
        except OSError:
            pass
        return "The Maia runtime differs from this player's training run. Restore it or retrain before playing."
    return None


def public_state(profile):
    state = profile.state
    board = chess.Board()
    history = []
    timeline = [{"fen": board.fen(), "last_move": None, "in_check": board.is_check()}]
    for uci in state["moves"]:
        move = board.parse_uci(uci)
        history.append(board.san(move))
        board.push(move)
        timeline.append({"fen": board.fen(), "last_move": uci, "in_check": board.is_check()})
    samples = state["policy"]["samples"]
    metrics = state["metrics"]
    live_samples = metrics.get("count", samples)
    game_over = state["result"] != "*" or board.is_game_over()
    legal_positions = {}
    if not game_over:
        for move in list(board.legal_moves):
            san = board.san(move)
            board.push(move)
            try:
                legal_positions[move.uci()] = {"fen": board.fen(), "last_move": move.uci(),
                                               "in_check": board.is_check(), "san": san}
            finally:
                board.pop()
    status = engine_room().status()
    error = runtime_error(state)
    if error:
        status.update(ready=False, engine_error=error)
    return {"revision": profile.revision, "fen": board.fen(), "moves": history, "timeline": timeline,
            "legal_moves": list(legal_positions), "legal_positions": legal_positions,
            "last_move": state["moves"][-1] if state["moves"] else None,
            "in_check": board.is_check(), "game_over": game_over, "result": state["result"],
            "turn": "white" if board.turn else "black", "mode": state["mode"],
            "events": state["events"][-50:], "archive": state["archive"], "decision": state["decision"],
            "profile": {"samples": samples, "live_samples": live_samples, "username": state.get("username"),
                        "training": state.get("training"), "games": state["games"], "completed": state["completed"],
                        "mean_loss": round(state["total_loss"] / live_samples) if live_samples else None,
                        "prior_log_loss": metrics["prior_loss"] / live_samples if live_samples else None,
                        "personal_log_loss": metrics["personal_loss"] / live_samples if live_samples else None,
                        "prior_hits": metrics["prior_hits"], "personal_hits": metrics["personal_hits"]},
            **status}


@ensure_csrf_cookie
@require_GET
def chat_view(request):
    profile_for(request)
    return render(request, "chat.html")


@require_GET
def state_view(request):
    return JsonResponse(public_state(profile_for(request)))


def apply_move(state, text):
    error = runtime_error(state)
    if error:
        raise EngineUnavailable(error)
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
    if not state.get("policy_fingerprint"):
        state["policy_fingerprint"] = current_maia_fingerprint()
    policy = PersonalPolicy(state["policy"])
    san = board.san(move)
    move_number = board.fullmove_number
    with engine_room().turn() as room:
        observation = room.observe(board, move, policy)
        board.push(move)
        state["moves"].append(move.uci())
        event = {**observation, "number": move_number, "player": san, "mode": state["mode"],
                 "adapted": False}
        state["decision"] = None
        if not board.is_game_over():
            reply, decision = room.choose(board, policy, state["mode"])
            event["opponent"] = board.san(reply)
            event["adapted"] = decision["personal_changed"]
            event["engine_changed"] = decision["engine_changed"]
            event["search_cost_cp"] = decision["engine_cost_cp"]
            event["decision"] = decision
            state["decision"] = decision
            board.push(reply)
            state["moves"].append(reply.uci())
    state["policy"] = policy.dump()
    state["total_loss"] += observation["loss_cp"]
    metrics = state["metrics"]
    metrics["count"] = metrics.get("count", policy.samples - 1) + 1
    metrics["prior_loss"] += observation["prior_log_loss"]
    metrics["personal_loss"] += observation["personal_log_loss"]
    metrics["prior_hits"] += int(observation["prior_hit"])
    metrics["personal_hits"] += int(observation["personal_hit"])
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
    state.update(moves=[], events=[], result="*", mode=mode, games=state["games"] + 1, decision=None)


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
                if profile.state.get("username"):
                    state["username"] = profile.state["username"]
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
    except (EngineUnavailable, MaiaUnavailable) as error:
        return JsonResponse({"error": str(error)}, status=503)
    except OperationalError:
        return JsonResponse({"error": "NEMESIS is busy. Please retry your move."}, status=409)
    except Exception:
        LOGGER.exception("NEMESIS action failed; transaction rolled back")
        return JsonResponse({"error": "The move could not be completed. Your saved position is unchanged."}, status=500)


@require_GET
def export_view(request):
    profile = profile_for(request)
    if request.GET.get("format") == "json":
        from .chat_views import transcript
        response = JsonResponse({"version": 2, "state": profile.state, "chat": transcript(profile, limit=None)}, json_dumps_params={"indent": 2})
        response["Content-Disposition"] = 'attachment; filename="nemesis-research.json"'
        return response
    game = chess.pgn.Game.from_board(restore_board(profile.state))
    game.headers.update(Event="NEMESIS personal training", White="You", Black="NEMESIS",
                        Result=profile.state["result"], Mode=profile.state["mode"])
    response = HttpResponse(str(game), content_type="application/x-chess-pgn")
    response["Content-Disposition"] = 'attachment; filename="nemesis-game.pgn"'
    return response
