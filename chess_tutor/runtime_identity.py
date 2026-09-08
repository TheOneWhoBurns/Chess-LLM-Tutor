"""Identify the frozen prior used for training and live inference."""
from functools import lru_cache
import hashlib
from pathlib import Path
import shutil

from .maia_policy import MaiaPolicy, POLICY_QUANTIZATION_VERSION


@lru_cache(maxsize=16)
def _file_hash(path, modified_ns, changed_ns, size):
    # Stat fields invalidate the cached hash when an installed artifact changes.
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _file_hash(str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


def current_maia_fingerprint():
    model = MaiaPolicy()
    binary = shutil.which(model.engine_path) or model.engine_path
    return {"cache_version": 2, "quantization": POLICY_QUANTIZATION_VERSION,
            "weights_sha256": _digest(model.weights_path),
            "engine_sha256": _digest(binary), "backend": model.backend,
            "wrapper_sha256": _digest(Path(__file__).with_name("maia_policy.py"))}
