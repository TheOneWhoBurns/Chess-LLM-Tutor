"""Session-scoped practice API. All graded state lives on the server."""
import json
import logging
import uuid

from django.db import OperationalError, transaction
from django.db.models import F
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods

from .models import PracticeAttempt, PracticeLesson
from .opponent import EngineUnavailable
from .practice import attempt_public, extract_saved_mistakes, finish_attempt, summary
from .views import profile_for

LOGGER = logging.getLogger(__name__)


@ensure_csrf_cookie
@require_GET
def practice_page(request):
    profile_for(request)
    return render(request, 'practice.html')


def start(profile):
    active = PracticeAttempt.objects.filter(lesson__profile=profile, finished_at__isnull=True).first()
    if active:
        return active
    lesson = profile.practice_lessons.filter(due_at__lte=timezone.now()).first()
    if lesson is None:
        return None
    # SQLite's select_for_update is a no-op. Write before re-reading to serialize
    # starts without touching the live game's revision or learned policy.
    PracticeLesson.objects.filter(pk=lesson.pk).update(version=F('version') + 1)
    lesson.refresh_from_db()
    active = lesson.attempts.filter(finished_at__isnull=True).first()
    if active:
        return active
    if lesson.due_at > timezone.now():
        return None
    return PracticeAttempt.objects.create(lesson=lesson,
        is_review=lesson.attempts.filter(finished_at__isnull=False).exists())


@require_http_methods(['GET', 'POST'])
def practice_api(request):
    profile = profile_for(request)
    if request.method == 'GET':
        active = PracticeAttempt.objects.filter(lesson__profile=profile, finished_at__isnull=True).first()
        if active is None:
            active = PracticeAttempt.objects.filter(lesson__profile=profile).order_by('-started_at').first()
        return JsonResponse({'summary': summary(profile), 'attempt': attempt_public(active) if active else None})
    try:
        data = json.loads(request.body)
        if not isinstance(data, dict):
            raise ValueError('Send a JSON object.')
        action = data.get('action')
        if action not in ('extract', 'start', 'hint', 'submit', 'reveal'):
            raise ValueError('Unknown practice action.')
        with transaction.atomic():
            if action == 'extract':
                extracted = extract_saved_mistakes(profile)
                return JsonResponse({'extracted': extracted, 'summary': summary(profile)})
            if action == 'start':
                attempt = start(profile)
            else:
                try:
                    attempt_id = uuid.UUID(str(data.get('id', '')))
                except ValueError:
                    raise ValueError('Choose an active lesson.') from None
                attempt = PracticeAttempt.objects.filter(pk=attempt_id, lesson__profile=profile).first()
                if attempt is None:
                    return JsonResponse({'error': 'Lesson not found.'}, status=404)
                if attempt.finished_at:
                    return JsonResponse({'attempt': attempt_public(attempt), 'summary': summary(profile)})
                revision = data.get('revision')
                if type(revision) is not int:
                    raise ValueError('Refresh the lesson before answering.')
                changed = PracticeAttempt.objects.filter(pk=attempt.pk, revision=revision,
                    finished_at__isnull=True).update(revision=F('revision') + 1)
                if not changed:
                    attempt.refresh_from_db()
                    return JsonResponse({'error': 'This lesson changed in another tab. Review it and retry.',
                                         'attempt': attempt_public(attempt)}, status=409)
                attempt.refresh_from_db()
                if action == 'hint':
                    attempt.hint_level = min(2, attempt.hint_level + 1)
                    attempt.save(update_fields=['hint_level'])
                else:
                    move = data.get('move', '')
                    reasoning = data.get('reasoning', '')
                    if not isinstance(move, str) or len(move) > 16:
                        raise ValueError('Enter a move in SAN or UCI notation.')
                    if not isinstance(reasoning, str) or len(reasoning) > 2000:
                        raise ValueError('Keep your explanation under 2,000 characters.')
                    finish_attempt(attempt, move, reasoning.strip(), reveal=action == 'reveal')
            return JsonResponse({'attempt': attempt_public(attempt) if attempt else None,
                                 'summary': summary(profile)})
    except (ValueError, UnicodeDecodeError) as error:
        return JsonResponse({'error': str(error)}, status=400)
    except EngineUnavailable as error:
        return JsonResponse({'error': str(error)}, status=503)
    except OperationalError:
        return JsonResponse({'error': 'Practice is busy. Please retry.'}, status=409)
    except Exception:
        LOGGER.exception('Practice request failed; transaction rolled back')
        return JsonResponse({'error': 'Practice could not complete. Please retry.'}, status=500)
