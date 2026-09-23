# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Serialize read/render/write operations for one clip within the API process."""

from contextlib import contextmanager
from functools import wraps
import threading


_registry_lock = threading.Lock()
_locks = {}


@contextmanager
def clip_operation(clip_id: str):
    # Count holders AND waiters so an entry cannot disappear while another
    # caller still owns its lock. Unrelated clips remain independent.
    with _registry_lock:
        entry = _locks.setdefault(clip_id, [threading.RLock(), 0])
        entry[1] += 1
    acquired = False
    try:
        entry[0].acquire()
        acquired = True
        yield
    finally:
        if acquired:
            entry[0].release()
        with _registry_lock:
            entry[1] -= 1
            if entry[1] == 0:
                del _locks[clip_id]


def serialized_clip_operation(operation):
    """Hold ownership across a ClipService operation's DB read and write."""
    @wraps(operation)
    def wrapped(self, clip_id, *args, **kwargs):
        with clip_operation(clip_id):
            return operation(self, clip_id, *args, **kwargs)
    return wrapped
