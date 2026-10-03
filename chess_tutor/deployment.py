"""Activate a completed local history fit without overwriting live player data."""
import hashlib
import json
import math
from pathlib import Path

from django.db import transaction

from .models import PlayerProfile
from .personal_profile import PLAYER_CONFIG, canonical_username, named_profile_id
from .player_policy import PersonalPolicy
from .runtime_identity import current_maia_fingerprint
from .views import initial_state


def activate_player(username, training_dir, config_path=None):
    username = canonical_username(username)
    directory = Path(training_dir)
    model_bytes = (directory / "deployment_model.json").read_bytes()
    model = json.loads(model_bytes)
    report = json.loads((directory / "report.json").read_text())
    summary = report["summary"]
    if model.get("complete_dataset") is not True or report.get("training_complete") is not True or \
            model.get("purpose") != "all_selected_games_for_live_play":
        raise ValueError("Only a completed fit on the full imported dataset can be activated.")
    if any(canonical_username(value) != username for value in
           (model.get("username"), summary.get("username"))):
        raise ValueError("The trained model belongs to a different player.")
    for key in ("dataset_fingerprint", "policy_fingerprint"):
        if not model.get(key) or model[key] != report.get(key):
            raise ValueError("The deployment model and evaluation report do not match.")
    if model["policy_fingerprint"] != current_maia_fingerprint():
        raise ValueError("This model was trained with a different Maia runtime. Restore that runtime or retrain.")
    policy = PersonalPolicy(model["policy"])
    for key in ("games_total", "train_games", "test_games", "train_positions", "test_positions", "total_positions"):
        if type(summary.get(key)) is not int or summary[key] < 0:
            raise ValueError("The training report has invalid counts.")
    if summary["total_positions"] != policy.samples or not policy.samples or \
            summary["train_positions"] + summary["test_positions"] != policy.samples or \
            summary["train_games"] + summary["test_games"] != summary["games_total"]:
        raise ValueError("The model observation count does not match the completed training report.")
    for key in ("prior_log_loss", "personal_log_loss", "prior_accuracy", "personal_accuracy"):
        value = summary.get(key)
        if value is None and not summary["test_positions"]:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or \
                (key.endswith("accuracy") and value > 1):
            raise ValueError("The training report has invalid evaluation metrics.")
    digest = hashlib.sha256(model_bytes).hexdigest()
    state = initial_state()
    state.update(username=username, policy=policy.dump(), training=dict(summary),
                 deployment_sha256=digest, policy_fingerprint=model["policy_fingerprint"])
    with transaction.atomic():
        profile, created = PlayerProfile.objects.get_or_create(
            pk=named_profile_id(username), defaults={"state": state})
        if not created and profile.state.get("deployment_sha256") != digest:
            raise ValueError("This player already has a different saved model. Export it before replacing it.")
    # Activating the same artifact again only restores this pointer; it never
    # rolls back games or neural updates collected after the original activation.
    config = Path(config_path) if config_path is not None else PLAYER_CONFIG
    config.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.with_suffix(".tmp")
    temporary.write_text(json.dumps({"username": username}, indent=2) + "\n")
    temporary.replace(config)
    return {"username": username, "profile_id": str(profile.pk), "created": created,
            "samples": profile.state["policy"]["samples"], "config": str(config)}
