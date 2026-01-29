#!/usr/bin/env python3
"""
Setup script for Maia chess engine dependencies.

Downloads Maia neural network weights and verifies LC0 installation.
"""

import os
import sys
import shutil
import subprocess
import urllib.request
from pathlib import Path

# Maia weights download URL (from official CSSLab repository)
MAIA_WEIGHTS = {
    "maia-1100": "https://github.com/CSSLab/maia-chess/raw/master/maia_weights/maia-1100.pb.gz",
    "maia-1200": "https://github.com/CSSLab/maia-chess/raw/master/maia_weights/maia-1200.pb.gz",
    "maia-1300": "https://github.com/CSSLab/maia-chess/raw/master/maia_weights/maia-1300.pb.gz",
    "maia-1400": "https://github.com/CSSLab/maia-chess/raw/master/maia_weights/maia-1400.pb.gz",
    "maia-1500": "https://github.com/CSSLab/maia-chess/raw/master/maia_weights/maia-1500.pb.gz",
    "maia-1600": "https://github.com/CSSLab/maia-chess/raw/master/maia_weights/maia-1600.pb.gz",
    "maia-1700": "https://github.com/CSSLab/maia-chess/raw/master/maia_weights/maia-1700.pb.gz",
    "maia-1800": "https://github.com/CSSLab/maia-chess/raw/master/maia_weights/maia-1800.pb.gz",
    "maia-1900": "https://github.com/CSSLab/maia-chess/raw/master/maia_weights/maia-1900.pb.gz",
}

DEFAULT_WEIGHT = "maia-1100"


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def check_lc0_installed():
    """Check if LC0 is installed and accessible."""
    print("Checking for LC0 installation...")

    lc0_path = shutil.which("lc0")
    if lc0_path:
        print(f"  ✓ LC0 found at: {lc0_path}")
        return True

    print("  ✗ LC0 not found in PATH")
    print("\n  To install LC0:")
    print("    macOS:   brew install lc0")
    print("    Ubuntu:  sudo apt install lc0")
    print("    Windows: Download from https://github.com/LeelaChessZero/lc0/releases")
    print("    Or build from source: https://github.com/LeelaChessZero/lc0")
    return False


def download_weights(weight_name=DEFAULT_WEIGHT, force=False):
    """Download Maia weights if not present."""
    project_root = get_project_root()
    weights_dir = project_root / "maia-chess" / "maia_weights"
    weights_file = weights_dir / f"{weight_name}.pb.gz"

    # Create directory if needed
    weights_dir.mkdir(parents=True, exist_ok=True)

    if weights_file.exists() and not force:
        print(f"  ✓ Weights already exist: {weights_file}")
        return True

    url = MAIA_WEIGHTS.get(weight_name)
    if not url:
        print(f"  ✗ Unknown weight: {weight_name}")
        print(f"    Available: {', '.join(MAIA_WEIGHTS.keys())}")
        return False

    print(f"  Downloading {weight_name} weights...")
    print(f"    From: {url}")
    print(f"    To: {weights_file}")

    try:
        urllib.request.urlretrieve(url, weights_file, _download_progress)
        print(f"\n  ✓ Downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        return False


def _download_progress(block_num, block_size, total_size):
    """Show download progress."""
    if total_size > 0:
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        sys.stdout.write(f"\r    [{bar}] {percent:.1f}%")
        sys.stdout.flush()


def verify_setup():
    """Verify the complete setup."""
    project_root = get_project_root()
    weights_file = project_root / "maia-chess" / "maia_weights" / f"{DEFAULT_WEIGHT}.pb.gz"

    print("\n" + "=" * 50)
    print("SETUP VERIFICATION")
    print("=" * 50)

    lc0_ok = check_lc0_installed()
    weights_ok = weights_file.exists()

    if weights_ok:
        print(f"  ✓ Maia weights present")
    else:
        print(f"  ✗ Maia weights missing")

    print("\n" + "-" * 50)
    if lc0_ok and weights_ok:
        print("✓ Setup complete! You can now run the chess tutor.")
        print("\n  Start with: python manage.py runserver")
        return True
    else:
        print("✗ Setup incomplete. Please fix the issues above.")
        return False


def main():
    """Main setup routine."""
    import argparse

    parser = argparse.ArgumentParser(description="Setup Maia chess engine")
    parser.add_argument(
        "--weight",
        default=DEFAULT_WEIGHT,
        choices=list(MAIA_WEIGHTS.keys()),
        help=f"Which Maia weight to download (default: {DEFAULT_WEIGHT})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of weights"
    )
    parser.add_argument(
        "--all-weights",
        action="store_true",
        help="Download all available Maia weights"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("MAIA CHESS ENGINE SETUP")
    print("=" * 50)
    print()

    # Check LC0
    lc0_ok = check_lc0_installed()
    print()

    # Download weights
    print("Checking Maia weights...")
    if args.all_weights:
        for weight in MAIA_WEIGHTS:
            download_weights(weight, args.force)
    else:
        download_weights(args.weight, args.force)

    # Verify
    verify_setup()


if __name__ == "__main__":
    main()
