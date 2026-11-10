"""Atomic output files (SPEC ENG-7).

Every output is written to a temporary file in the same directory and then renamed onto its final name
(`os.replace`, atomic on POSIX file systems), so an interrupted run never leaves a half-written file under a final
name. A failed write removes its temporary file and re-raises.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np


@contextlib.contextmanager
def replacing(path, mode: str = "w"):
    """Open a temporary file next to `path` for writing; on a clean exit it replaces `path`."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    f = open(tmp, mode)
    try:
        yield f
        f.close()
        os.replace(tmp, path)
    except BaseException:
        f.close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def write_text(path, text: str) -> None:
    with replacing(path, "w") as f:
        f.write(text)


def savetxt(path, array, **kwargs) -> None:
    """np.savetxt to `path`, atomically."""
    with replacing(path, "w") as f:
        np.savetxt(f, array, **kwargs)


def savez_compressed(path, **arrays) -> None:
    """np.savez_compressed to `path` (the name as given; ".npz" is not appended), atomically."""
    with replacing(path, "wb") as f:
        np.savez_compressed(f, **arrays)
