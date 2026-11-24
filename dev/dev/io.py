"""Reading and writing cached analysis results."""

import os
import tempfile

import numpy as np


def load_npz(path, prefix=""):
    """All arrays in an ``.npz`` whose key starts with ``prefix``, prefix removed.

    Zero-dimensional arrays come back as Python scalars.
    """
    with np.load(path) as data:
        return {
            key.removeprefix(prefix): data[key].item() if data[key].ndim == 0 else data[key]
            for key in data.files
            if key.startswith(prefix)
        }


def save_npz_atomic(path, **arrays):
    """Write a compressed ``.npz`` so that readers never see a half-written file."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=f".{stem}.", suffix=".npz", delete=False
    ) as temporary:
        temporary_path = temporary.name
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def cache_matches(path, keys=(), **expected):
    """Whether ``path`` holds all ``keys`` and stored values close to ``expected``."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    with np.load(path) as cached:
        return set(keys) | set(expected) <= set(cached.files) and all(
            np.shape(cached[name]) == np.shape(value) and np.allclose(cached[name], value)
            for name, value in expected.items()
        )
