"""Persistent, browser-scoped NEMESIS player state."""
import uuid

from django.db import models


class PlayerProfile(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    state = models.JSONField(default=dict)
    revision = models.PositiveIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)
