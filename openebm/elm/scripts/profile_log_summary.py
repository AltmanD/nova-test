#!/usr/bin/env python3
"""
Parse profiling logs emitted by run_ebt_2node_8gpu_profile.sh and convert the
four [profile_*] line families into:

1. step-level CSV
2. stage-level CSV (front / middle / late)
3. Markdown summary for quick comparison

Usage:
    python openebm/elm/scripts/profile_log_summary.py --log /path/to/train.log

Optional:
    --out_dir /path/to/output_dir
    --label run_name_for_report
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean


PROFILE_PREFIXES = (
    "profile_step",
    "profile_timing",
    "profile_data",
    "profile_state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize EBT profiling logs into CSV and Markdown.")
    parser.add_argument("--log", required=True, help="Path to the profiling training log file.")
    parser.add_argument(
        "--out_dir",
        default="",
        help="Output directory. Defaults to <log_dir>/<log_stem>_profile_summary",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional label shown in the Markdown report header.",
    )
    return parser.parse_args()


def to_number(raw: str):
    raw = raw.strip().rstrip(",")
    if raw.lower() in {"nan", "none"}:
        return math.nan
    if raw.startswith("+"):
        raw = raw[1:]
    raw_no_commas = raw.replace(",", "")
    if re.fullmatch(r"-?\d+", raw_no_commas):
        return int(raw_no_commas)
    try:
        return float(raw_no_commas)
    except ValueError:
        return raw


def parse_profile_line(line: str):
    match = re.search(r"\[(profile_[a-z]+)\]\s+(.*)", line)
    if not match:
        return None
    prefix = match.group(1)
    payload = match.group(2).strip()
    fields = {}
    for token in payload.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = to_number(value)
    return prefix, fields


def load_steps(log_path: Path):
    by_step = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            parsed = parse_profile_line(raw_line)
            if parsed is None:
                continue
            prefix, fields = parsed
            step_value = fields.get("step")
            if step_value is None:
                continue
            step_key = int(str(step_value).split("/")[0]) if isinstance(step_value, str) else int(step_value)
            entry = by_step.setdefault(step_key, {"step": step_key})
            for key, value in fields.items():
                if key == "step" and isinstance(value, str) and "/" in value:
                    cur, total = value.split("/", 1)
                    entry["step"] = int(cur)
                    entry["max_steps"] = int(total)
                else:
                    entry[key] = value
            entry["profile_family"] = prefix
    steps = [by_step[k] for k in sorted(by_step.keys())]
    return steps


def safe_mean(values):
    filtered = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return mean(filtered) if filtered else math.nan


def safe_min(values):
    filtered = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return min(filtered) if filtered else math.nan


def safe_max(values):
    filtered = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return max(filtered) if filtered else math.nan


def split_stages(steps):
    if not steps:
        return []
    n = len(steps)
    base = n // 3
    rem = n % 3
    sizes = [base, base, base]
    for i in range(rem):
        sizes[i] += 1
    labels = ["front", "middle", "late"]
    result = []
    start = 0
    for label, size in zip(labels, sizes):
        end = start + size
        chunk = steps[start:end]
        if chunk:
            result.append((label, chunk))
        start = end
    return result


SUMMARY_KEYS = [
    "tok_s",
    "step_ms",
    "wait_avg_ms",
    "wait_max_ms",
    "fw_avg_ms",
    "bw_avg_ms",
    "opt_ms",
    "dl_avg_ms",
    "dl_max_ms",
    "fetch_avg_ms",
    "tok_avg_ms",
    "tok_max_ms",
    "pack_avg_ms",
    "pack_max_ms",
    "copy_avg_ms",
    "pq",
    "rg",
    "doc_batch",
    "docbuf_end",
    "docbuf_max",
    "refill_calls",
    "packed_docs",
    "cropped_docs",
    "packed_tok",
    "cropped_tok",
]


def summarize_stage(label, steps):
    row = {
        "stage": label,
        "num_optimizer_steps": len(steps),
        "step_start": steps[0]["step"],
        "step_end": steps[-1]["step"],
    }
    for key in SUMMARY_KEYS:
        values = [s.get(key, math.nan) for s in steps]
        row[f"{key}_avg"] = safe_mean(values)
        row[f"{key}_min"] = safe_min(values)
        row[f"{key}_max"] = safe_max(values)
    return row


def write_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=2):
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown(label: str, log_path: Path, steps, stage_rows):
    title = label or log_path.stem
    lines = []
    lines.append(f"# Profiling Summary: {title}")
    lines.append("")
    lines.append(f"- Source log: `{log_path}`")
    lines.append(f"- Parsed optimizer steps: `{len(steps)}`")
    if steps:
        lines.append(f"- Step range: `{steps[0]['step']}` -> `{steps[-1]['step']}`")
    lines.append("")

    if not stage_rows:
        lines.append("No `[profile_*]` records were found.")
        lines.append("")
        return "\n".join(lines)

    lines.append("## Stage Comparison")
    lines.append("")
    lines.append("| Stage | Step Range | tok/s avg | step_ms avg | wait_avg_ms avg | tok_avg_ms avg | pack_avg_ms avg | cropped_docs avg | cropped_tok avg |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in stage_rows:
        lines.append(
            "| {stage} | {start}-{end} | {tok_s} | {step_ms} | {wait_ms} | {tok_ms} | {pack_ms} | {cropped_docs} | {cropped_tok} |".format(
                stage=row["stage"],
                start=row["step_start"],
                end=row["step_end"],
                tok_s=fmt(row["tok_s_avg"], 0),
                step_ms=fmt(row["step_ms_avg"]),
                wait_ms=fmt(row["wait_avg_ms_avg"]),
                tok_ms=fmt(row["tok_avg_ms_avg"]),
                pack_ms=fmt(row["pack_avg_ms_avg"]),
                cropped_docs=fmt(row["cropped_docs_avg"]),
                cropped_tok=fmt(row["cropped_tok_avg"]),
            )
        )
    lines.append("")

    front = next((r for r in stage_rows if r["stage"] == "front"), None)
    late = next((r for r in stage_rows if r["stage"] == "late"), None)
    if front and late:
        lines.append("## Front vs Late Delta")
        lines.append("")
        lines.append("| Metric | Front Avg | Late Avg | Delta |")
        lines.append("| --- | ---: | ---: | ---: |")
        for key in ("tok_s", "step_ms", "wait_avg_ms", "tok_avg_ms", "pack_avg_ms", "cropped_docs", "cropped_tok"):
            fval = front[f"{key}_avg"]
            lval = late[f"{key}_avg"]
            delta = lval - fval if all(isinstance(x, (int, float)) and not math.isnan(x) for x in (fval, lval)) else math.nan
            lines.append(
                f"| {key} | {fmt(fval)} | {fmt(lval)} | {fmt(delta)} |"
            )
        lines.append("")

    lines.append("## Reading Guide")
    lines.append("")
    lines.append("- `tok_s` down + `step_ms` up: overall throughput is getting worse.")
    lines.append("- `wait_avg_ms` up: GPU waits longer for the next micro-batch.")
    lines.append("- `tok_avg_ms` up: tokenizer work is becoming more expensive.")
    lines.append("- `pack_avg_ms` up: best-fit packing is becoming more expensive.")
    lines.append("- `cropped_docs` / `cropped_tok` up: the current shard distribution is forcing more cropping.")
    lines.append("- `pq` / `rg` progression can be aligned with throughput drops to locate problematic shard ranges.")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    log_path = Path(args.log).resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = log_path.parent / f"{log_path.stem}_profile_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = load_steps(log_path)
    if not steps:
        raise RuntimeError("No [profile_*] lines found in the provided log.")

    step_csv = out_dir / "profile_steps.csv"
    stage_csv = out_dir / "profile_stage_summary.csv"
    report_md = out_dir / "profile_summary.md"

    write_csv(step_csv, steps)

    stage_rows = [summarize_stage(label, chunk) for label, chunk in split_stages(steps)]
    write_csv(stage_csv, stage_rows)

    report_md.write_text(build_markdown(args.label, log_path, steps, stage_rows), encoding="utf-8")

    print(f"Parsed steps: {len(steps)}")
    print(f"Step CSV: {step_csv}")
    print(f"Stage CSV: {stage_csv}")
    print(f"Markdown summary: {report_md}")


if __name__ == "__main__":
    main()
