"""Build the comparison tables from measured artifacts.

The README's result cells are filled from this script rather than typed, so a number in
the README always traces back to a file a run produced. A table transcribed by hand is a
table that drifts from its evidence.

Missing results print as TBD rather than being omitted, so it stays obvious which parts of
the comparison have actually been measured.

    python compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

ARTIFACTS = Path(__file__).parent / "artifacts"

ACCURACY_ROWS = [
    ("keyword + position rules", "0", "words", "project1/baseline_metrics.json"),
    ("LayoutLMv3-base (Project 1)", "125M", "words + boxes + page image", "project1/metrics.json"),
    ("Qwen2.5-7B-Instruct, few-shot", "7B", "words only", "base_metrics.json"),
    ("Qwen2.5-7B-Instruct, QLoRA", "7B", "words only", "qlora_metrics.json"),
]


def load(name: str) -> dict | None:
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def cell(value, spec: str = "{:.3f}") -> str:
    return spec.format(value) if isinstance(value, (int, float)) else "TBD"


def accuracy_table() -> str:
    rows = [
        "| model | params | inputs | F1 | parse rate | locate rate |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for label, params, inputs, source in ACCURACY_ROWS:
        data = load(source) or {}
        f1 = cell(data.get("test_f1"))
        rows.append(
            f"| {label} | {params} | {inputs} | **{f1}** | "
            f"{cell(data.get('parse_rate'), '{:.2f}') if 'parse_rate' in data else '-'} | "
            f"{cell(data.get('locate_rate'), '{:.2f}') if 'locate_rate' in data else '-'} |"
        )
    return "\n".join(rows)


ENGINE_TITLES = {
    "hf": "### transformers, batch size 1",
    "vllm": "### vLLM (separate session - compare within this table, not against the one above)",
}


def _short(error: str, limit: int = 90) -> str:
    """One line, bounded. A multi-line exception pasted into a cell destroys the table."""
    first = error.split("\n")[0].strip()
    return first if len(first) <= limit else first[: limit - 1] + "…"


def efficiency_table() -> str:
    sections = []
    for engine in ("hf", "vllm"):
        report = load(f"bench_{engine}.json")
        if not report:
            continue
        rows = [
            ENGINE_TITLES[engine],
            "",
            "| config | workload | load s | p50 ms | p95 ms | tok/s | peak VRAM GB |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for result in report["results"]:
            if "error" in result:
                rows.append(
                    f"| {result['config']} | **not measured** - {_short(result['error'])} "
                    "| - | - | - | - | - |"
                )
                continue
            rows.append(
                f"| {result['config']} | {result['workload']} | {result['load_seconds']} | "
                f"{result['warm_latency_p50_ms']} | {result['warm_latency_p95_ms']} | "
                f"{result['tokens_per_second'] or '-'} | "
                f"{result['peak_vram_gb'] if result.get('peak_vram_gb') is not None else 'n/a'} |"
            )
        sections.append("\n".join(rows))

    if not sections:
        return (
            "| config | workload | load s | p50 ms | p95 ms | tok/s | peak VRAM GB |\n"
            "|---|---|---:|---:|---:|---:|---:|\n"
            "| _not yet measured_ | - | TBD | TBD | TBD | TBD | TBD |"
        )
    return "\n\n".join(sections)


def ceiling_table() -> str:
    report = load("converter_ceiling.json")
    rows = ["| target format | ceiling F1 | used |", "|---|---:|---|"]
    flat = cell((report or {}).get("flat_entity_list", {}).get("f1"))
    nested = cell((report or {}).get("nested_gt_parse", {}).get("f1"))
    rows.append(f"| flat entity list, label word included | {flat} | yes |")
    rows.append(f"| nested `gt_parse`, value only | {nested} | no |")
    return "\n".join(rows)


def main() -> None:
    truncation = (load("qlora_metrics.json") or {}).get("truncation_rate")
    overlap = (load("split_check.json") or {}).get("train_test_overlap")

    document = "\n\n".join(
        [
            "## Accuracy",
            accuracy_table(),
            "## Efficiency",
            efficiency_table(),
            "## Converter ceiling",
            ceiling_table(),
            "## Checks",
            "\n".join(
                [
                    f"- train/test word-sequence overlap: "
                    f"{overlap if overlap is not None else 'TBD'}",
                    f"- truncated generations (QLoRA): "
                    f"{cell(truncation, '{:.2%}') if truncation is not None else 'TBD'}",
                ]
            ),
        ]
    )

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "comparison.md").write_text(document, encoding="utf-8")
    print(document)
    print(f"\nwritten to {ARTIFACTS / 'comparison.md'}")


if __name__ == "__main__":
    main()
