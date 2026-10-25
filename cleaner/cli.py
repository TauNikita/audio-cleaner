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
                   help="output audio file; format inferred from extension (default: clean.wav); "
                        "the cuts are logged next to it as <output>.cuts.json")
    p.add_argument("--dry-run", action="store_true",
                   help="align and print the report only; write no audio")
    p.add_argument("--clauses", action="store_true",
                   help="align clauses (split at commas, semicolons, colons and dashes) instead of "
                        "whole sentences; picks up retakes from the last comma but the extra cuts "
                        "can clip word endings")
    p.add_argument("--no-refine", action="store_true",
                   help="skip the listening passes: the rescue of units the transcript could not "
                        "place and the hidden-retake pass inside each chosen span")
    p.add_argument("--overrides", metavar="JSON",
                   help="hand placements for stubborn units: a JSON object mapping unit number to "
                        "{\"runs\": [[start, end], ...]} in seconds of the recording; those units "
                        "are cut exactly so and skipped by the listening passes")
    p.add_argument("--verify", action="store_true",
                   help="after rendering, re-transcribe each sentence from the output and flag "
                        "any that do not match the script or run long")

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

    g = p.add_argument_group("assembly (ms)",
                             "Pieces are cut at their audible edges. A pause is the silence heard between "
                             "two pieces, from the last sound of one to the first sound of the next.")
    g.add_argument("--sentence-pause", type=int, default=600,
                   help="pause heard between sentences (default: 600)")
    g.add_argument("--paragraph-pause", type=int, default=600,
                   help="pause heard between paragraphs (default: 600)")
    g.add_argument("--clause-pause", type=int, default=300,
                   help="pause heard inside a sentence: between clauses (--clauses), at a shortened "
                        "inner pause and at a join where a retake was cut out (default: 300)")
    g.add_argument("--lead", type=int, default=100,
                   help="room kept before a piece's first sound (default: 100)")
    g.add_argument("--tail", type=int, default=80,
                   help="room kept after a piece's last sound has died down (default: 80)")
    g.add_argument("--tail-max", type=int, default=900,
                   help="how far past the last word's timestamp the sound may run on before the "
                        "cut is forced (default: 900)")
    g.add_argument("--max-inner-pause", type=int, default=700,
                   help="a silence inside a sentence longer than this is shortened to the clause "
                        "pause; 0 keeps every inner pause as recorded (default: 700)")
    g.add_argument("--max-internal-gap", type=int, default=1200,
                   help="a gap between consecutive transcript words longer than this splits the "
                        "span, dropping the dead air or retake in between (default: 1200)")
    g.add_argument("--hard-fade", type=int, default=40,
                   help="fade at a cut that has to land in running speech (default: 40)")
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
        sentences = split_script(fh.read(), clauses=args.clauses)
    if not sentences:
        print("error: the script has no sentences", file=sys.stderr)
        return 2
    if not looks_like_prose(sentences):
        print("warning: the script looks like bullet points or shorthand rather than full "
              "sentences; matching will be poor. Full sentences work best.", file=sys.stderr)
    print(f"Script: {len(sentences)} units ({sum(1 for s in sentences if s.sentence_end)} sentences, "
          f"{sum(1 for s in sentences if s.paragraph_end)} paragraphs).")

    from . import transcribe as tr
    from .align import align
    from .report import print_report
    from .audio import DependencyError

    cache_path = args.cache or tr.default_cache_path(args.media, args.model)

    # The 16kHz WAV feeds Whisper, the listening passes and the edge finder of the assembly;
    # rendering cuts from the original media. A warm cache on a dry run touches neither ffmpeg nor
    # Whisper.
    need_transcribe = args.refresh or tr.load_cache(
        cache_path, args.media, args.model, args.language) is None
    refine_on = not args.no_refine
    need_wav = need_transcribe or refine_on or args.verify or not args.dry_run

    wav_path = None
    is_temp_wav = False
    try:
        # Resolve device/compute only when a transcribe is actually on the table.
        if need_transcribe or refine_on or args.verify:
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

        energy = prober = None
        if refine_on or args.verify:
            from .hidden import Energy, Prober, refine
            energy = Energy(wav_path)
            prober = Prober(tr._load_model(args.model, args.device, args.compute_type), args.language)
        if args.overrides:
            import json
            from .align import MATCH
            with open(args.overrides, "r", encoding="utf-8") as fh:
                overrides = json.load(fh)
            for key, spec in overrides.items():
                d = decisions[int(key) - 1]
                runs = [(float(a), float(b)) for a, b in spec["runs"]]
                d.start, d.end, d.runs, d.locked = runs[0][0], runs[-1][1], runs, True
                d.status, d.score, d.excise = MATCH, 100.0, []
                d.notes.append("placed by hand (--overrides)")
                if d.word_start is None:
                    d.word_start = d.word_end = next((i for i, w in enumerate(words) if w.start >= d.start), len(words))
            print(f"Overrides: {len(overrides)} unit(s) placed by hand.")
        if refine_on:
            from .rescue import rescue
            rescue(decisions, words, energy, prober, threshold=args.threshold, max_extra=args.max_extra)
            refine(decisions, words, energy, prober, max_gap_s=args.max_internal_gap / 1000.0)
        counts = print_report(decisions, threshold=args.threshold, model=args.model)

        if args.dry_run:
            print("Dry run: no audio written. Re-run without --dry-run to render.")
            return 0

        if energy is None:
            from .hidden import Energy
            energy = Energy(wav_path)
        from .assemble import assemble
        positions = assemble(args.media, decisions, words, args.output, energy,
                             lead_ms=args.lead, tail_ms=args.tail, tail_max_ms=args.tail_max,
                             sentence_pause_ms=args.sentence_pause, paragraph_pause_ms=args.paragraph_pause,
                             clause_pause_ms=args.clause_pause, max_internal_gap_ms=args.max_internal_gap,
                             max_inner_pause_ms=args.max_inner_pause, hard_fade_ms=args.hard_fade)
        if args.verify:
            from .audio import extract_wav
            from .verify import verify
            out_wav = extract_wav(args.output)
            try:
                verify(out_wav, positions, prober)
            finally:
                if os.path.exists(out_wav):
                    os.remove(out_wav)
        return 0

    except (DependencyError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if is_temp_wav and wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
