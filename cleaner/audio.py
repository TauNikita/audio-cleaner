"""Audio extraction via ffmpeg.

Both transcription and cutting reference the exact same 16kHz mono WAV, so timestamps from Whisper
line up precisely with the samples we later slice.
"""

import os
import platform
import re
import shutil
import subprocess
import tempfile

from .util import Progress, fmt_ts


class DependencyError(RuntimeError):
    """Raised when an external tool (ffmpeg) is missing, with an install hint."""


def _ffmpeg_install_hint() -> str:
    system = platform.system()
    if system == "Windows":
        return ("winget install Gyan.FFmpeg   (then open a NEW terminal so it lands on PATH)\n"
                "  or:  choco install ffmpeg   /   scoop install ffmpeg")
    if system == "Darwin":
        return "brew install ffmpeg"
    return "sudo apt install ffmpeg"


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise DependencyError(
            f"ffmpeg not found on PATH. Install it with:\n  {_ffmpeg_install_hint()}\n"
            "If you just installed it, open a new terminal (PATH changes don't reach the current one)."
        )


def probe_duration(media: str) -> float | None:
    """Best-effort media duration in seconds (used only to show extract progress)."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", media],
            capture_output=True, text=True, timeout=60,
        )
        return float(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return None


_OUT_TIME = re.compile(r"out_time_ms=(\d+)")


def extract_wav(media: str) -> str:
    """Extract the audio track to a temp 16kHz mono WAV. Returns the temp path.

    The caller owns the returned file and must delete it (do this in a finally block).
    """
    ensure_ffmpeg()
    fd, wav_path = tempfile.mkstemp(prefix="audio_cleaner_", suffix=".wav")
    # mkstemp opens the file; close the descriptor so ffmpeg can write to the path.
    os.close(fd)

    duration = probe_duration(media)
    progress = Progress("Extract")
    progress.update("starting ffmpeg", force=True)

    cmd = [
        "ffmpeg", "-y", "-i", media,
        "-vn",                 # drop any video stream
        "-ac", "1",            # mono
        "-ar", "16000",        # 16kHz
        "-f", "wav",
        "-progress", "pipe:1", "-nostats", "-loglevel", "error",
        wav_path,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for line in proc.stdout:
            m = _OUT_TIME.search(line)
            if not m:
                continue
            done = int(m.group(1)) / 1_000_000  # microseconds -> seconds
            if duration:
                pct = min(100.0, done / duration * 100)
                progress.update(f"{pct:5.1f}%  {fmt_ts(done)} / {fmt_ts(duration)}")
            else:
                progress.update(fmt_ts(done))
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()

    if proc.returncode != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        try:
            os.remove(wav_path)
        except OSError:
            pass
        raise DependencyError(f"ffmpeg failed to extract audio:\n{stderr.strip()}")

    progress.done(f"done ({fmt_ts(duration) if duration else 'ok'})")
    return wav_path
