from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import MAX_VOCAB_SIZE, ROOT
from .data import load_dataset
from .evaluation import evaluate_submission
from .leaderboard import build_leaderboard, write_leaderboard, write_site_json
from .metrics import benchmark
from .validation import validate_tokenizer


def check_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an AMTC tokenizer submission")
    parser.add_argument("tokenizer", nargs="?", default="tokenizer.json")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--benchmark-data", default=str(ROOT / "tests/fixtures/validation.csv"))
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args(argv)
    report = validate_tokenizer(args.tokenizer)
    local_benchmark = None
    benchmark_path = Path(args.benchmark_data)
    if report.valid and report.tokenizer is not None and not args.no_benchmark and benchmark_path.is_file():
        examples = load_dataset(benchmark_path)
        throughput, elapsed = benchmark(report.tokenizer, examples, repeats=3)
        local_benchmark = {
            "data": str(benchmark_path),
            "elapsed_seconds": elapsed,
            "throughput_chars_per_second": throughput,
        }
    if args.as_json:
        payload = report.as_dict()
        payload["local_benchmark"] = local_benchmark
        print(json.dumps(payload, indent=2))
    else:
        print("African Multilingual Tokenizer Challenge\nSubmission Checker\n")
        labels = {
            "loads": "Loading tokenizer",
            "file_size": "File size",
            "vocabulary": "Vocabulary",
            "encodes_all_languages": "Six-language encoding",
            "decodes": "Decoding",
            "compatible_version": "Compatibility",
        }
        for key in ("loads", "file_size", "vocabulary", "encodes_all_languages", "decodes", "compatible_version"):
            if key in report.checks:
                detail = ""
                if key == "vocabulary" and report.vocab_size is not None:
                    detail = f" {report.vocab_size:,} / {MAX_VOCAB_SIZE:,}"
                print(f"{labels[key]:<26} {'✓' if report.checks[key] else '✗'}{detail}")
        if local_benchmark:
            print("\nLocal benchmark (informational)")
            print(f"Evaluation time            {local_benchmark['elapsed_seconds']:.4f} s")
            print(f"Throughput                 {local_benchmark['throughput_chars_per_second']:,.0f} characters/second")
        print("\nREADY FOR SUBMISSION ✓" if report.valid else "\nNOT READY FOR SUBMISSION")
        for error in report.errors:
            print(f"✗ {error}")
    return 0 if report.valid else 1


def evaluate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score one AMTC submission")
    parser.add_argument("submission_dir")
    parser.add_argument("--data", default="tests/fixtures/validation.csv")
    parser.add_argument("--baseline", default="tests/fixtures/baseline_fertility.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    result = evaluate_submission(args.submission_dir, args.data, args.baseline, benchmark_repeats=args.repeats)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def leaderboard_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate submissions and update AMTC leaderboards")
    parser.add_argument("--submissions", default="submissions")
    parser.add_argument("--data", default="tests/fixtures/demo_public_test.csv")
    parser.add_argument("--baseline", default="tests/fixtures/baseline_fertility.json")
    parser.add_argument("--csv", default="leaderboard.csv")
    parser.add_argument("--markdown", default="LEADERBOARD.md")
    parser.add_argument("--site-json", default="docs/data/leaderboard.json")
    parser.add_argument("--failures", default="artifacts/evaluation_failures.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--final-only", action="store_true", help="rank only entries with final: true")
    args = parser.parse_args(argv)
    rows, failures = build_leaderboard(
        args.submissions,
        args.data,
        args.baseline,
        benchmark_repeats=args.repeats,
        final_only=args.final_only,
    )
    write_leaderboard(rows, args.csv, args.markdown)
    write_site_json(rows, args.site_json)
    failures_path = Path(args.failures)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluated {len(rows)} valid submission(s); {len(failures)} failed.")
    for failure in failures:
        print(f"✗ {failure['slug']}: {failure['error']}")
    return 1 if args.strict and failures else 0
