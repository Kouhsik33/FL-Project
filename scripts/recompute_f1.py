"""Correct F1 in saved summaries (bug fix of 2026-10-20; see docs/final_report.md §7).

Before the fix, F1 was reported as undefined (None) whenever a model predicted no
positives, even if the evaluated set contained positives; the correct value is
F1 = 2TP / (2TP + FP + FN) = 0. Every saved metrics dict stores its confusion
counts, so F1 is recomputed exactly from them. Nothing else in the metrics changes;
per-client disparity and local-only means are recomputed from the corrected values.

Usage: python scripts/recompute_f1.py [--dry-run]
Originals must be archived first (results/archive/summaries_before_f1_fix.tar.gz).
"""

import argparse
import json

import numpy as np

from fedmed.evaluation.metrics import clean_for_json, disparity
from fedmed.experiments.run import DISPARITY_METRICS
from fedmed.paths import RESULTS_DIR, RUNS_DIR

TAG = "f1_recomputed_from_confusion_counts_2026-10-20"


def fix(obj, changes):
    if isinstance(obj, dict):
        if {"tp", "fp", "fn", "f1"} <= obj.keys():
            denom = 2 * obj["tp"] + obj["fp"] + obj["fn"]
            new = (2 * obj["tp"] / denom) if denom > 0 else None
            old = obj["f1"]
            if (old is None) != (new is None) or (old is not None and abs(old - new) > 1e-9):
                changes.append((old, new))
            obj["f1"] = new
        for v in obj.values():
            fix(v, changes)
    elif isinstance(obj, list):
        for v in obj:
            fix(v, changes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not (RESULTS_DIR / "archive" / "summaries_before_f1_fix.tar.gz").exists():
        raise SystemExit("Archive the original summaries first")
    total_changed, runs_changed = 0, 0
    for path in sorted(RUNS_DIR.glob("*/summary.json")):
        s = json.loads(path.read_text())
        if TAG in s.get("corrections", []):
            continue
        changes: list = []
        fix(s, changes)
        final = s["final"]
        if s["mode"] == "local":
            for split, key in (("test", "global_test"), ("val", "val")):
                vals = [m[key]["f1"] for m in final["local_models"] if m[key]["f1"] is not None]
                final[split]["f1"] = float(np.mean(vals)) if vals else None
        final["disparity"] = {m: disparity(final["client_test"], m) for m in DISPARITY_METRICS}
        s.setdefault("corrections", []).append(TAG)
        total_changed += len(changes)
        runs_changed += bool(changes)
        if not args.dry_run:
            path.write_text(json.dumps(clean_for_json(s), indent=1))
    print(f"{runs_changed} runs had F1 values corrected ({total_changed} metric dicts); dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
