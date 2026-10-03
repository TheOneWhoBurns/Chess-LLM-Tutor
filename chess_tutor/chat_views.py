"""Persistent Astra coaching without changing the board or personal policy."""
from datetime import timedelta
import json
import logging
import uuid

from django.db import IntegrityError, OperationalError, transaction
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from . import astra
from .models import ChatExchange
from .opponent import EngineUnavailable, engine_room
from .views import profile_for, restore_board

LOGGER = logging.getLogger(__name__)
MAX_MESSAGE = 4000


class ChatConflict(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def transcript(profile, limit=50):
    rows = profile.chat_exchanges.filter(status="complete").order_by("-id")
    rows = list(rows[:limit] if limit is not None else rows)
    messages = []
    for row in reversed(rows):
        for role, content in (("user", row.user_message), ("assistant", row.assistant_message)):
            messages.append({"role": role, "content": content, "context_label": row.context_label,
                             "model": row.model if role == "assistant" else None})
    return messages


def payload(profile, error=None, error_code=None):
    result = {**astra.configuration_status(), "messages": transcript(profile), "revision": profile.revision}
    if error:
        result.update(error=error, error_code=error_code)
    return result


def context_label(state):
    board = restore_board(state)
    if state["result"] != "*":
        return f"Game {state['games']} · Result {state['result']}"
    return f"Game {state['games']} · Move {board.fullmove_number} · {'White' if board.turn else 'Black'} to play"


def fresh_analysis(state):
    board = restore_board(state)
    if state["result"] != "*" or board.is_game_over():
        return {"game_over": True, "result": state["result"] if state["result"] != "*" else board.result()}
    try:
        with engine_room().turn() as room:
            rows = room.stockfish.rank(board, count=3, nodes=60000)
            return {"engine": room.stockfish.name, "perspective": "white" if board.turn else "black",
                    "moves": [{"move": board.san(row["move"]), "score_cp": row["score"],
                               "mate": row["mate"], "depth": row["depth"], "nodes": row["nodes"]}
                              for row in rows]}
    except EngineUnavailable:
        return {"unavailable": True, "error": "Fresh engine analysis is unavailable."}


def claim_exchange(profile, request_id, message, revision):
    """Reserve one request in a short transaction, before any external call."""
    with transaction.atomic():
        # A crashed worker must not block this local player's chat indefinitely.
        profile.chat_exchanges.filter(status="pending", updated_at__lt=timezone.now() - timedelta(seconds=120)).update(
            status="failed", error="The previous request was interrupted.", updated_at=timezone.now())
        existing = profile.chat_exchanges.filter(request_id=request_id).first()
        if existing:
            if existing.user_message != message or existing.board_revision != revision:
                raise ChatConflict("This request identifier was already used for another message.", "request_conflict")
            if existing.status == "complete":
                return existing, False
            if existing.status == "pending":
                raise ChatConflict("Astra is still answering this message. Retry shortly.", "chat_busy")
        if profile.chat_exchanges.filter(status="pending").exists():
            raise ChatConflict("Astra is still answering your previous message. Retry shortly.", "chat_busy")
        if existing:
            existing.status, existing.error = "pending", ""
            existing.save(update_fields=["status", "error", "updated_at"])
            exchange = existing
        else:
            exchange = ChatExchange.objects.create(profile=profile, request_id=request_id,
                user_message=message, board_revision=revision, context_label="")
        # The write reservation also serializes this read against SQLite game
        # updates. The later API call holds no database transaction or game lock.
        profile.refresh_from_db()
        if profile.revision != revision:
            raise ChatConflict("The board changed. Review the current position and send your message again.", "stale_revision")
        exchange.context_label = context_label(profile.state)
        exchange.save(update_fields=["context_label", "updated_at"])
        return exchange, True


@require_http_methods(["GET", "POST"])
def chat_api(request):
    profile = profile_for(request)
    if request.method == "GET":
        return JsonResponse(payload(profile))
    exchange = None
    try:
        data = json.loads(request.body)
        if not isinstance(data, dict):
            raise ValueError("Send a JSON object.")
        message = data.get("message")
        if not isinstance(message, str) or not 1 <= len(message.strip()) <= MAX_MESSAGE:
            raise ValueError(f"Enter a message of 1–{MAX_MESSAGE:,} characters.")
        message = message.strip()
        revision = data.get("revision")
        if type(revision) is not int or revision < 0:
            raise ValueError("The current board revision is required.")
        if not isinstance(data.get("request_id"), str):
            raise ValueError("A request identifier is required.")
        try:
            request_id = uuid.UUID(data["request_id"])
        except ValueError as error:
            raise ValueError("The request identifier is invalid.") from error
        # Completed retries can be served locally even if API access has changed.
        saved = profile.chat_exchanges.filter(request_id=request_id, status="complete").first()
        if saved:
            if saved.user_message != message or saved.board_revision != revision:
                raise ChatConflict("This request identifier was already used for another message.", "request_conflict")
            return JsonResponse(payload(profile))
        status = astra.configuration_status()
        if not status["ready"]:
            return JsonResponse(payload(profile, status["error"], "not_configured"), status=503)
        exchange, claimed = claim_exchange(profile, request_id, message, revision)
        if not claimed:
            return JsonResponse(payload(profile))
        evidence = fresh_analysis(profile.state)
        context = astra.build_context(profile.state, revision, evidence)
        response = astra.answer(message, transcript(profile, limit=10), context)
        # The timestamp identifies this claim, so an expired worker cannot write
        # over a newer attempt using the same request identifier.
        updated = ChatExchange.objects.filter(pk=exchange.pk, status="pending", updated_at=exchange.updated_at).update(
            assistant_message=response, context=context, status="complete", updated_at=timezone.now())
        if not updated:
            raise ChatConflict("This chat request expired. Please send it again.", "request_expired")
        profile.refresh_from_db()
        return JsonResponse(payload(profile))
    except (ValueError, UnicodeDecodeError) as error:
        code, message, status = "invalid_request", str(error), 400
    except ChatConflict as error:
        code, message, status = error.code, str(error), 409
    except astra.AstraUnavailable as error:
        code, message, status = error.code, str(error), error.status_code
    except (OperationalError, IntegrityError):
        code, message, status = "chat_busy", "NEMESIS is busy. Please retry your message.", 409
    except Exception:
        LOGGER.exception("Astra chat request failed")
        code, message, status = "internal_error", "The chat could not complete this request. Please retry.", 500
    try:
        if exchange is not None:
            ChatExchange.objects.filter(pk=exchange.pk, status="pending", updated_at=exchange.updated_at).update(
                status="failed", error=message, updated_at=timezone.now())
        profile.refresh_from_db()
        return JsonResponse(payload(profile, message, code), status=status)
    except OperationalError:
        # A concurrent chess turn can still hold SQLite's write lock. The claim
        # expires if cleanup cannot run; omit messages so the UI retains its copy.
        return JsonResponse({**astra.configuration_status(), "error": message, "error_code": code}, status=status)
