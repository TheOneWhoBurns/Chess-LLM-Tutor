"""Persistent NEMESIS player state for a local identity or browser session."""
import uuid

from django.db import models
from django.utils import timezone


class PlayerProfile(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    state = models.JSONField(default=dict)
    revision = models.PositiveIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)


class ChatExchange(models.Model):
    """One idempotent coaching request, stored separately from chess learning."""
    profile = models.ForeignKey(PlayerProfile, on_delete=models.CASCADE, related_name="chat_exchanges")
    request_id = models.UUIDField()
    board_revision = models.PositiveIntegerField()
    context_label = models.CharField(max_length=120)
    context = models.JSONField(default=dict)
    user_message = models.TextField()
    assistant_message = models.TextField(blank=True)
    model = models.CharField(max_length=80, default="gpt-6-astra")
    status = models.CharField(max_length=12, default="pending")
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created_at", "id"]
        constraints = [
            models.UniqueConstraint(fields=["profile", "request_id"], name="nemesis_chat_request_id"),
            models.UniqueConstraint(fields=["profile"], condition=models.Q(status="pending"),
                                    name="nemesis_chat_one_pending"),
        ]


class PracticeLesson(models.Model):
    """A confirmed mistake and private engine evidence, independent of live play."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    profile = models.ForeignKey(PlayerProfile, on_delete=models.CASCADE, related_name="practice_lessons")
    source_key = models.CharField(max_length=64)
    evidence = models.JSONField(default=dict)
    due_at = models.DateTimeField(default=timezone.now)
    stage = models.PositiveSmallIntegerField(default=0)
    version = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["due_at", "created_at"]
        constraints = [models.UniqueConstraint(fields=["profile", "source_key"],
                                               name="nemesis_practice_source")]


class PracticeAttempt(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    lesson = models.ForeignKey(PracticeLesson, on_delete=models.CASCADE, related_name="attempts")
    revision = models.PositiveIntegerField(default=0)
    hint_level = models.PositiveSmallIntegerField(default=0)
    is_review = models.BooleanField(default=False)
    reasoning = models.TextField(blank=True)
    result = models.JSONField(default=dict)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["started_at"]
        constraints = [models.UniqueConstraint(fields=["lesson"], condition=models.Q(finished_at__isnull=True),
                                               name="nemesis_practice_one_active")]


class PracticeEvaluation(models.Model):
    """Remember checked positions, including mistakes rejected by deeper search."""
    profile = models.ForeignKey(PlayerProfile, on_delete=models.CASCADE, related_name="practice_evaluations")
    source_key = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["profile", "source_key"],
                                               name="nemesis_practice_evaluated")]
