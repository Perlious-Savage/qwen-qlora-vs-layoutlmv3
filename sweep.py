"""LoRA hyperparameter sweep, tracked in MLflow.

Same harness as Project 1's sweep, pointed at the parameters that are specific to this
project: LoRA rank and learning rate. A single run tells you what one configuration
scored; it does not tell you whether the configuration mattered. The sensitivity report
answers that, and it is also the cheapest defence against reading a 0.01 difference as a
result when it is inside seed noise.

Sweep runs never overwrite artifacts/qlora_metrics.json. That file holds the canonical
result from the reference configuration; a sweep is exploration.

Rank costs almost nothing in training time and a little in adapter size, so it gets the
grid points. Learning rate is held at one value by default because with only three
configurations, varying two things at once tells you about neither.

    python sweep.py --quick     # plumbing check, minutes
    python sweep.py --fast      # 3 ranks, 1 epoch
    python sweep.py             # 3 ranks, 2 epochs, roughly 3 hours on an A100
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ARTIFACTS = Path(__file__).parent / "artifacts"

DEFAULT_GRID = {"rank": [8, 16, 32], "lr": [2e-4], "epochs": [2.0]}

# Scoring a generative model means generating, which is slow. 40 validation documents is
# enough to rank configurations against each other; the canonical number comes from the
# full test split, scored separately.
SWEEP_EVAL_DOCUMENTS = 40


def sensitivity(results: list[dict], key: str) -> dict:
    """How much does the score move across the values of one hyperparameter?"""
    groups: dict[float, list[float]] = {}
    for record in results:
        groups.setdefault(record[key], []).append(record["validation_f1"])
    means = {value: statistics.mean(scores) for value, scores in groups.items()}
    if len(means) < 2:
        return {"values": {str(k): round(v, 4) for k, v in means.items()}, "spread": 0.0}
    return {
        "values": {str(k): round(v, 4) for k, v in sorted(means.items())},
        "spread": round(max(means.values()) - min(means.values()), 4),
        "best": str(max(means, key=means.get)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny grid, for checking the plumbing")
    parser.add_argument("--fast", action="store_true", help="3 ranks at 1 epoch instead of 2")
    args = parser.parse_args()

    from src.train_qlora import Config, run_training

    if args.quick:
        grid = {"rank": [8, 16], "lr": [2e-4], "epochs": [1.0]}
    elif args.fast:
        grid = {"rank": [8, 16, 32], "lr": [2e-4], "epochs": [1.0]}
    else:
        grid = DEFAULT_GRID

    combinations = [dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())]
    print(f"{len(combinations)} configurations\n")

    results = []
    for index, combo in enumerate(combinations, 1):
        name = f"r{combo['rank']}-lr{combo['lr']:g}-ep{combo['epochs']:g}"
        print(f"[{index}/{len(combinations)}] {name}")
        started = time.time()

        metrics = run_training(
            Config(
                rank=combo["rank"],
                # Keeping alpha at 2x rank holds the effective LoRA scaling constant, so
                # the sweep measures rank rather than rank-and-scaling together.
                alpha=combo["rank"] * 2,
                lr=combo["lr"],
                epochs=combo["epochs"],
                run_name=name,
                max_train=40 if args.quick else 0,
                eval_limit=8 if args.quick else SWEEP_EVAL_DOCUMENTS,
                # Exploration must not clobber the canonical result or the saved adapter.
                write_artifacts=False,
                save_model=False,
            )
        )

        results.append(
            {
                **combo,
                "run_name": name,
                "validation_f1": metrics["validation_f1"],
                "validation_parse_rate": metrics["validation_parse_rate"],
                "validation_locate_rate": metrics["validation_locate_rate"],
                "trainable_parameters": metrics["trainable_parameters"],
                "minutes": round((time.time() - started) / 60, 1),
            }
        )
        print(f"    F1 {metrics['validation_f1']:.4f}  ({results[-1]['minutes']} min)\n")

    results.sort(key=lambda r: -r["validation_f1"])
    best, worst = results[0], results[-1]

    summary = {
        "n_configurations": len(results),
        "eval_documents": SWEEP_EVAL_DOCUMENTS,
        "eval_split": "validation",
        "results": results,
        "best": best,
        "spread_across_grid": round(best["validation_f1"] - worst["validation_f1"], 4),
        "sensitivity": {
            "lora_rank": sensitivity(results, "rank"),
            "learning_rate": sensitivity(results, "lr"),
        },
        "note": (
            "One seed per configuration, scored on a validation subset. Differences "
            "smaller than run-to-run seed variance should not be read as real."
        ),
    }

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "sweep_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = [
        "| rank | lr | epochs | val F1 | parse | locate | trainable params | min |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ] + [
        f"| {r['rank']} | {r['lr']:g} | {r['epochs']:g} | **{r['validation_f1']:.4f}** | "
        f"{r['validation_parse_rate']:.2f} | {r['validation_locate_rate']:.2f} | "
        f"{r['trainable_parameters']:,} | {r['minutes']} |"
        for r in results
    ]
    table = "\n".join(rows)
    (ARTIFACTS / "sweep_table.md").write_text(table, encoding="utf-8")

    print(table)
    print(f"\nbest: {best['run_name']} at F1 {best['validation_f1']:.4f}")
    print(f"spread across the grid: {summary['spread_across_grid']:.4f}")
    print(f"rank sensitivity: {summary['sensitivity']['lora_rank']}")
    print(f"\nwritten to {ARTIFACTS / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
