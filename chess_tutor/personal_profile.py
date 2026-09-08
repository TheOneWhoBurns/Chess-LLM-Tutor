"""Identity for an optional, locally configured one-person installation."""
import json
import logging
from pathlib import Path
import re
import uuid


PLAYER_CONFIG = Path(__file__).resolve().parents[1] / ".runtime/player.json"
LOGGER = logging.getLogger(__name__)


def canonical_username(username):
    if not isinstance(username, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{2,50}", username):
        raise ValueError("Enter a valid Chess.com username.")
    return username.casefold()


def named_profile_id(username):
    return uuid.uuid5(uuid.NAMESPACE_URL, "https://www.chess.com/member/" + canonical_username(username))


def active_username(config_path=None):
    path = Path(config_path) if config_path is not None else PLAYER_CONFIG
    try:
        return canonical_username(json.loads(path.read_text())["username"])
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        LOGGER.warning("Cannot read the local player configuration at %s", path)
        return None
