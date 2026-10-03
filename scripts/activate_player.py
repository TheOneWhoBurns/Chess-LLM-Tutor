#!/usr/bin/env python3
"""Select a completed history model as the local player's shared profile."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chess_tutor.settings")

import django
django.setup()

from chess_tutor.deployment import activate_player


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--training-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = activate_player(args.username, args.training_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Activation failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
