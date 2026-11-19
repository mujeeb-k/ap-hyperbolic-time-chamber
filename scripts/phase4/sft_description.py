"""The SFT data description (the pre-registered plan, Phase 6, specified 2026-11-02): the distribution of tool calls and
tokens per demonstration, by case type, from the frozen sidecar of sft-demos/v2-1 (its sha256 checked against
manifests/sft_demos_v2_1/FREEZE.json). Laptop only: no model and no GPU. The demonstrations and their freeze record
are not part of this release, so the script does not run from the release alone.

Tool calls are the assistant's calls in the demonstration, the planted mistake's included. A planted malformed call never
reaches the environment, so here it counts one more than the replay's environment calls (27 of the 300 demonstrations).
Tokens are the rendering's: completion (everything after the prompt, tool responses included), context (prompt plus
completion, the whole row) and trained (the tokens the labels keep).

    .venv/bin/python scripts/phase4/sft_description.py              # prints the table and writes description.json
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "runs" / "phase4" / "sft-demos-v2-1"
RECORD = ROOT / "manifests" / "sft_demos_v2_1" / "FREEZE.json"
TYPES = ("V1", "V2", "V3a", "V3b", "V4", "V5")
MEASURES = {"tool_calls": "tool calls", "completion_tokens": "completion tokens", "context_tokens": "context tokens",
            "trained_tokens": "trained tokens"}


def demonstrations(sidecar: pathlib.Path = DATA / "sidecar.jsonl") -> list[dict]:
    """The sidecar's rows, refused unless the file is the frozen one."""
    record = json.loads(RECORD.read_text())
    (key,) = [k for k in record if k.startswith("dataset_sha256")]
    got = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    if got != record[key]["sidecar.jsonl"]:
        raise SystemExit(f"{sidecar} is not the frozen sidecar: sha256 {got}")
    return [json.loads(line) for line in sidecar.open()]


def measures(row: dict) -> dict[str, int]:
    render = row["checks"]["render"]
    return {"tool_calls": sum(len(m.get("tool_calls") or []) for m in row["messages"]),
            "completion_tokens": render["completion_tokens"], "context_tokens": render["context_tokens"],
            "trained_tokens": render["trained_tokens"]}


def distribution(values: list[int]) -> dict:
    q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else (values[0],) * 3
    return {"min": min(values), "q1": q1, "median": median, "q3": q3, "max": max(values),
            "mean": round(statistics.fmean(values), 1)}


def describe(rows: list[dict]) -> dict:
    """Per case type and overall: the five-number summary and mean of each measure, and the count of each call total."""
    def group(subset: list[dict]) -> dict:
        values = [measures(r) for r in subset]
        calls = [v["tool_calls"] for v in values]
        return {"demonstrations": len(subset), **{k: distribution([v[k] for v in values]) for k in MEASURES},
                "tool_call_counts": {str(c): calls.count(c) for c in sorted(set(calls))}}
    return {"dataset": "sft-demos/v2-1", "by_type": {t: group([r for r in rows if r["type"] == t]) for t in TYPES},
            "all": group(rows)}


def table(description: dict) -> str:
    lines = ["| Type | Demonstrations | " + " | ".join(f"{v} (min / Q1 / median / Q3 / max; mean)" for v in MEASURES.values())
             + " |", "|---|---|" + "---|" * len(MEASURES)]
    for name, g in [*description["by_type"].items(), ("All", description["all"])]:
        cells = []
        for k in MEASURES:
            d = g[k]
            spec = ".4g" if k == "tool_calls" else ",.0f"                # tokens to the whole token, calls with their .5
            cells.append(" / ".join(format(d[q], spec) for q in ("min", "q1", "median", "q3", "max"))
                         + f"; {format(d['mean'], '.1f' if k == 'tool_calls' else ',.0f')}")
        lines.append(f"| {name} | {g['demonstrations']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    description = describe(demonstrations())
    (DATA / "description.json").write_text(json.dumps(description, indent=1))
    print(table(description))
    print("\nTool-call totals:")
    for name, g in [*description["by_type"].items(), ("All", description["all"])]:
        print(f"- {name}: " + ", ".join(f"{c} calls: {n}" for c, n in g["tool_call_counts"].items()))


if __name__ == "__main__":
    main()
