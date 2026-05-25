from __future__ import annotations

import os
import time

_START_TIME = time.monotonic()


def log_progress(stage: str, message: str, **fields):
    elapsed = time.monotonic() - _START_TIME
    parts = [
        "[ASRO_PROGRESS]",
        f"elapsed={elapsed:.1f}s",
        f"pid={os.getpid()}",
        f"stage={stage}",
        message,
    ]
    for key, value in fields.items():
        if value is not None:
            parts.append(f"{key}={value}")
    print(" | ".join(parts), flush=True)
