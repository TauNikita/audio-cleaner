"""Small shared helpers: timestamp formatting and single-line progress output."""

import sys
import time


def fmt_ts(seconds: float) -> str:
    """Format seconds as H:MM:SS.mmm (or M:SS.mmm under an hour)."""
    if seconds is None:
        return "--:--"
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:d}:{s:02d}.{ms:03d}"


class Progress:
    """Throttled single-line progress printer.

    Rewrites one terminal line so long stages (extract, transcribe, align, assemble) show movement
    without scrolling. Falls back to plain newlines when stderr isn't a TTY (logs, pipes).
    """

    def __init__(self, label: str, stream=sys.stderr, min_interval: float = 0.1):
        self.label = label
        self.stream = stream
        self.min_interval = min_interval
        self.is_tty = hasattr(stream, "isatty") and stream.isatty()
        self._last = 0.0
        self._width = 0

    def update(self, message: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last) < self.min_interval:
            return
        self._last = now
        line = f"{self.label}: {message}"
        if self.is_tty:
            pad = max(0, self._width - len(line))
            self.stream.write("\r" + line + " " * pad)
            self._width = len(line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def done(self, message: str) -> None:
        line = f"{self.label}: {message}"
        if self.is_tty:
            pad = max(0, self._width - len(line))
            self.stream.write("\r" + line + " " * pad + "\n")
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
