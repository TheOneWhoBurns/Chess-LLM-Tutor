#!/usr/bin/env python3
"""Install the pinned, verified Stockfish/Maia runtime locally on macOS.

LC0 is a separate prerequisite. No model/engine is copied into the Git repo's
tracked source, and no system files or shell startup configuration are changed.
"""
import argparse
import hashlib
from pathlib import Path
import platform
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SF_URL = 'https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-macos-universal.tar.gz'
SF_SHA = 'a1f0e3bcc5a6927a11fe6fc8e54a779754645f3c2bae2cf13420fd1957adaa77'
MAIA_URL = 'https://github.com/CSSLab/maia-chess/releases/download/v1.0/maia-1500.pb.gz'
MAIA_SHA = '35ab6f20421d59e1df3b17c5a5016947af4c6761368ef84044a9a9c7619a9a00'


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def asset(url, path, expected, verify_only):
    if not path.exists():
        if verify_only:
            raise SystemExit(f'Missing runtime asset: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.download')
        print(f'Downloading {path.name} from its official GitHub release…', flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output)
            if digest(temporary) != expected:
                raise SystemExit(f'Checksum mismatch for {path.name}; installation stopped.')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if digest(path) != expected:
        raise SystemExit(f'Checksum mismatch for {path}; installation stopped.')
    print(f'Verified {path.name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Check existing pinned downloads without network access')
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        raise SystemExit('This installer is for macOS. Install Stockfish and LC0 for your platform and set NEMESIS_STOCKFISH / NEMESIS_LC0_ENGINE; see README.md.')
    archive = ROOT / '.runtime/stockfish/stockfish-macos-universal.tar.gz'
    weights = ROOT / '.runtime/maia/maia-1500.pb.gz'
    asset(SF_URL, archive, SF_SHA, args.verify_only)
    asset(MAIA_URL, weights, MAIA_SHA, args.verify_only)
    binary = archive.parent / 'stockfish/stockfish-macos-universal'
    if not args.verify_only:
        with tarfile.open(archive) as source:
            source.extractall(archive.parent, filter='data')
    if not binary.is_file():
        raise SystemExit('Stockfish has not been extracted. Run without --verify-only.')
    if not shutil.which('lc0'):
        print('LC0 is still required: install it from https://lczero.org/play/quickstart/ or set NEMESIS_LC0_ENGINE.')
    print('Project runtime ready. Start Django with python manage.py runserver 127.0.0.1:8765.')


if __name__ == '__main__':
    main()
