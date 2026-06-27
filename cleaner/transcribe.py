"""Word-level transcription via faster-whisper, with a JSON cache.

Transcribing a multi-hour recording on CPU is slow, and the alignment threshold usually needs a few
tries to dial in. Caching the word list keyed on the media file + model means you pay that cost once
and every later dry-run is instant. `--refresh` forces a re-transcribe.
"""

import importlib
import json
import os
import platform
import re
from dataclasses import asdict, dataclass

from .audio import DependencyError
from .text import normalize
from .util import Progress, fmt_ts

CACHE_VERSION = 1


def cuda_device_count() -> int:
    """Number of CUDA devices ctranslate2 can see (0 if none / not a CUDA build)."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


def resolve_device(device: str, compute_type: str | None) -> tuple[str, str]:
    """Resolve the requested device and pick a compute type to match.

    - "auto": use the GPU when one is visible, otherwise fall back to CPU.
    - "cuda": require a visible GPU; fail loudly with a hint if there isn't one.
    - "cpu": as asked.

    When --compute-type isn't given we default to float16 on GPU and int8 on CPU.
    """
    device = (device or "auto").lower()
    count = cuda_device_count()

    if device == "auto":
        device = "cuda" if count > 0 else "cpu"
        if device == "cpu":
            print("Device: no CUDA GPU detected, using cpu.")
    elif device == "cuda" and count == 0:
        raise DependencyError(
            "CUDA was requested (--device cuda) but no usable GPU was detected.\n"
            "Check that the NVIDIA driver is installed (run `nvidia-smi`) and that ctranslate2\n"
            "has CUDA support plus the cuBLAS/cuDNN libraries (see the GPU section of the README).\n"
            "Use --device cpu to run on the processor instead."
        )

    if not compute_type:
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


@dataclass
class Word:
    raw: str        # the word as transcribed
    start: float    # seconds
    end: float      # seconds
    norm: str       # normalized form used for matching


def default_cache_path(media: str, model: str) -> str:
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return f"{media}.{safe_model}.words.json"


def _signature(media: str, model: str, language: str) -> dict:
    st = os.stat(media)
    return {
        "media": os.path.abspath(media),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "model": model,
        "language": language,
        "version": CACHE_VERSION,
    }


def load_cache(cache_path: str, media: str, model: str, language: str) -> list[Word] | None:
    """Return cached words if the cache exists and matches the current inputs, else None."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("meta") != _signature(media, model, language):
        return None
    return [Word(**w) for w in data.get("words", [])]


def save_cache(cache_path: str, media: str, model: str, language: str, words: list[Word]) -> None:
    payload = {"meta": _signature(media, model, language), "words": [asdict(w) for w in words]}
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, cache_path)


def _register_cuda_dll_dirs() -> None:
    """On Windows, make the pip-installed CUDA DLLs loadable by ctranslate2.

    The nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels drop their DLLs in
    site-packages\\nvidia\\<lib>\\bin, which isn't on the DLL search path, so ctranslate2 can't find
    cublas64_12.dll / cudnn*_9.dll. Registering those directories makes `uv sync --extra gpu` enough
    to run on the GPU with no manual PATH edits. No-op off Windows (Linux uses LD_LIBRARY_PATH).

    Both os.add_dll_directory and a PATH prepend are needed: Python's own loader honors
    add_dll_directory, but ctranslate2's native loader calls LoadLibrary with default search flags
    that ignore it and only consult PATH. Without the PATH entry the model constructs fine but the
    first GPU op dies with "cublas64_12.dll ... cannot be loaded".
    """
    if platform.system() != "Windows":
        return
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            mod = importlib.import_module(pkg)
        except ImportError:
            continue
        # These are namespace packages, so __file__ is None — use __path__ for the directory.
        bases = list(getattr(mod, "__path__", []) or [])
        if getattr(mod, "__file__", None):
            bases.append(os.path.dirname(mod.__file__))
        for base in bases:
            for sub in ("bin", "lib"):
                lib_dir = os.path.join(base, sub)
                if os.path.isdir(lib_dir):
                    try:
                        os.add_dll_directory(lib_dir)
                    except OSError:
                        pass
                    if lib_dir not in os.environ.get("PATH", "").split(os.pathsep):
                        os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")


def _cuda_lib_hint() -> str:
    if platform.system() == "Windows":
        return ("Install the GPU libraries with:  uv sync --extra gpu\n"
                "  (the tool then loads cuBLAS/cuDNN automatically — see the README GPU section).\n"
                "  Or run on the processor with:  --device cpu")
    return ("Install the GPU libraries with:  uv sync --extra gpu  and set LD_LIBRARY_PATH\n"
            "  (see the README GPU section). Or run on the processor with:  --device cpu")


def _is_cuda_lib_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("cudnn", "cublas", "cuda", "libcu", "cublas64"))


def _load_model(model: str, device: str, compute_type: str):
    if device == "cuda":
        _register_cuda_dll_dirs()  # before importing ctranslate2, in case it links CUDA at import
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is not installed. Install it with:\n"
            "  pip install faster-whisper"
        ) from exc
    try:
        return WhisperModel(model, device=device, compute_type=compute_type)
    except RuntimeError as exc:
        if device == "cuda" and _is_cuda_lib_error(exc):
            raise DependencyError(
                f"Failed to start the model on the GPU: {exc}\n{_cuda_lib_hint()}"
            ) from exc
        raise


def transcribe_words(
    wav_path: str,
    model: str,
    language: str,
    device: str,
    compute_type: str,
) -> list[Word]:
    """Transcribe the WAV into a flat, ordered list of words with timestamps."""
    progress = Progress("Transcribe")
    progress.update(f"loading model '{model}' ({device}/{compute_type})", force=True)
    whisper = _load_model(model, device, compute_type)

    segments, info = whisper.transcribe(
        wav_path,
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )
    total = getattr(info, "duration", None)

    words: list[Word] = []
    try:
        for segment in segments:  # generator: GPU compute happens lazily as we iterate
            for w in segment.words or []:
                norm = normalize(w.word)
                if not norm:
                    continue  # punctuation-only token, nothing to match against
                words.append(Word(raw=w.word.strip(), start=float(w.start),
                                  end=float(w.end), norm=norm))
            if total:
                pct = min(100.0, segment.end / total * 100)
                progress.update(f"{pct:5.1f}%  {fmt_ts(segment.end)} / {fmt_ts(total)} "
                                f"— {len(words)} words")
            else:
                progress.update(f"{fmt_ts(segment.end)} — {len(words)} words")
    except RuntimeError as exc:
        # CUDA libraries (cuBLAS/cuDNN) are loaded lazily on the first GPU op, so a missing-library
        # error often lands here rather than at model construction.
        if device == "cuda" and _is_cuda_lib_error(exc):
            raise DependencyError(
                f"GPU transcription failed: {exc}\n{_cuda_lib_hint()}"
            ) from exc
        raise

    progress.done(f"done — {len(words)} words")
    return words


def get_words(
    media: str,
    wav_path: str | None,
    model: str,
    language: str,
    device: str,
    compute_type: str,
    cache_path: str,
    refresh: bool,
) -> list[Word]:
    """Return words from cache when possible, otherwise transcribe and cache.

    `wav_path` may be None when a valid cache is expected; if a transcribe turns out to be needed
    without a WAV, that's a programming error in the caller.
    """
    if not refresh:
        cached = load_cache(cache_path, media, model, language)
        if cached is not None:
            print(f"Transcribe: using cached transcript ({len(cached)} words) at {cache_path}")
            return cached

    if wav_path is None:
        raise RuntimeError("transcription required but no WAV was extracted")

    words = transcribe_words(wav_path, model, language, device, compute_type)
    save_cache(cache_path, media, model, language, words)
    print(f"Transcribe: cached transcript to {cache_path}")
    return words
