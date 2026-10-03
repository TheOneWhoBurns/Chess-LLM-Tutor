"""Persistent NEMESIS player state for a local identity or browser session."""
import uuid

from django.db import models


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
