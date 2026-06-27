"""Command-line entry point.

Heavy dependencies (faster-whisper, pydub) and ffmpeg are only touched once we actually need them,
so `--help` and argument errors work with nothing installed. Missing pieces raise clear install
hints rather than tracebacks.
"""

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clean.py",
        description="Clean a messy voiceover so it follows a written script, "
                    "keeping the last good take of each sentence.",
    )
    p.add_argument("media", help="input recording (audio or video)")
    p.add_argument("script", help="script as a .txt file")
    p.add_argument("-o", "--output", default="clean.wav",
                   help="output audio file; format inferred from extension (default: clean.wav)")
    p.add_argument("--dry-run", action="store_true",
                   help="align and print the report only; write no audio")

    g = p.add_argument_group("transcription")
    g.add_argument("--model", default="small.en", help="faster-whisper model (default: small.en)")
    g.add_argument("--language", default="en", help="spoken language (default: en)")
    g.add_argument("--device", default="auto",
                   help="auto, cpu, or cuda (default: auto — GPU when available, else CPU)")
    g.add_argument("--compute-type", default=None,
                   help="ctranslate2 compute type, e.g. int8, float16, int8_float16 "
                        "(default: float16 on GPU, int8 on CPU)")
    g.add_argument("--cache", default=None, help="transcript cache path (default: next to media)")
    g.add_argument("--refresh", action="store_true", help="ignore the cache and re-transcribe")

    g = p.add_argument_group("alignment")
    g.add_argument("--threshold", type=float, default=82.0,
                   help="min fuzzy match score 0-100 (default: 82)")
    g.add_argument("--max-extra", type=int, default=3,
                   help="extra words to try beyond sentence length (default: 3)")
    g.add_argument("--horizon", type=int, default=None,
                   help="max words to scan ahead per sentence (default: unlimited)")

    g = p.add_argument_group("assembly (ms)")
    g.add_argument("--pad", type=int, default=60, help="padding added to each cut edge (default: 60)")
    g.add_argument("--sentence-pause", type=int, default=350,
                   help="silence between sentences (default: 350)")
    g.add_argument("--paragraph-pause", type=int, default=900,
                   help="silence between paragraphs (default: 900)")
    g.add_argument("--crossfade", type=int, default=15,
                   help="fade at each cut edge to avoid clicks (default: 15)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Light imports only; heavy ones stay inside the stages below.
    from .text import split_script, looks_like_prose

    if not os.path.exists(args.media):
        print(f"error: media file not found: {args.media}", file=sys.stderr)
        return 2
    if not os.path.exists(args.script):
        print(f"error: script file not found: {args.script}", file=sys.stderr)
        return 2

    with open(args.script, "r", encoding="utf-8") as fh:
        sentences = split_script(fh.read())
    if not sentences:
        print("error: the script has no sentences", file=sys.stderr)
        return 2
    if not looks_like_prose(sentences):
        print("warning: the script looks like bullet points or shorthand rather than full "
              "sentences; matching will be poor. Full sentences work best.", file=sys.stderr)
    print(f"Script: {len(sentences)} sentences, "
          f"{sum(1 for s in sentences if s.paragraph_end)} paragraphs.")

    from . import transcribe as tr
    from .align import align
    from .report import print_report
    from .audio import DependencyError

    cache_path = args.cache or tr.default_cache_path(args.media, args.model)

    # Extract the WAV only when we truly need it: a re-transcribe (cache miss or --refresh), or a
    # real render. A dry-run with a warm cache touches neither ffmpeg nor Whisper.
    need_transcribe = args.refresh or tr.load_cache(
        cache_path, args.media, args.model, args.language) is None
    need_wav = need_transcribe or not args.dry_run

    wav_path = None
    is_temp_wav = False
    try:
        # Resolve device/compute only when a transcribe is actually on the table.
        if need_transcribe:
            args.device, args.compute_type = tr.resolve_device(args.device, args.compute_type)
            print(f"Device: {args.device} / {args.compute_type}")

        if need_wav:
            from .audio import extract_wav
            wav_path = extract_wav(args.media)
            is_temp_wav = True

        words = tr.get_words(
            media=args.media, wav_path=wav_path, model=args.model, language=args.language,
            device=args.device, compute_type=args.compute_type,
            cache_path=cache_path, refresh=args.refresh,
        )
        if not words:
            print("error: transcription produced no words", file=sys.stderr)
            return 1

        decisions = align(words, sentences, threshold=args.threshold,
                          max_extra=args.max_extra, horizon=args.horizon)
        counts = print_report(decisions, threshold=args.threshold, model=args.model)

        if args.dry_run:
            print("Dry run: no audio written. Re-run without --dry-run to render.")
            return 0

        from .assemble import assemble
        assemble(wav_path, decisions, args.output, pad_ms=args.pad,
                 sentence_pause_ms=args.sentence_pause, paragraph_pause_ms=args.paragraph_pause,
                 crossfade_ms=args.crossfade)
        return 0

    except (DependencyError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if is_temp_wav and wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
