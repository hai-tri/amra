"""
Parity gates for TPU APRS runs.

The gates compare newly produced CSV rows against the existing Llama-3-8B GPU
reference artifacts under results/optuna_sweep and results/winner_validation.
They are intentionally metric-level gates: they catch pipeline drift between
TPU porting steps without requiring token-for-token equality from bf16/XLA.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Dict, Iterable, Optional


DEFAULT_REFERENCES = (
    "results/optuna_sweep/optuna_llama_*.csv",
    "results/optuna_sweep/nsga_llama_*.csv",
    "results/winner_validation/validate_llama_*.json",
)

METRIC_ALIASES = {
    "arditi_score_undefended": ("ref_undef_arditi",),
    "pca8_score_undefended": ("ref_undef_pca8",),
    "arditi_score_defended": ("ref_def_arditi",),
    "pca8_score_defended": ("ref_def_pca8",),
    "pile_bpb": ("bpb_def", "bpb"),
    "mmlu_acc": ("mmlu_def", "mmlu"),
    "math500_exact_match": ("math500_def", "math500"),
}


def _to_float(value) -> Optional[float]:
    if value in (None, "", "—"):
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if math.isnan(v):
        return None
    return v


def _load_csv_rows(path: str) -> Iterable[Dict]:
    with open(path, newline="") as f:
        yield from csv.DictReader(f)


def _load_reference_rows(patterns=DEFAULT_REFERENCES) -> list:
    rows = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            if path.endswith(".csv"):
                rows.extend(_load_csv_rows(path))
            elif path.endswith(".json"):
                with open(path) as f:
                    obj = json.load(f)
                model = obj.get("model", "")
                config = obj.get("config", {})
                def _mean(xs):
                    vals = [_to_float(x) for x in xs]
                    vals = [x for x in vals if x is not None]
                    return sum(vals) / len(vals) if vals else None
                rows.append({
                    "model": model,
                    "epsilon": config.get("epsilon"),
                    "num_layers": config.get("num_layers"),
                    "bpb": _mean(obj.get("defended", {}).get("bpb", [])),
                    "mmlu": _mean(obj.get("defended", {}).get("mmlu", [])),
                    "math500": _mean(obj.get("defended", {}).get("math500", [])),
                })
    return rows


def _score_match(candidate: Dict, reference: Dict) -> int:
    score = 0
    c_model = str(candidate.get("model", "")).lower()
    r_model = str(reference.get("model", "")).lower()
    if "llama" in c_model and "llama" in r_model:
        score += 3
    for key in ("epsilon", "num_layers"):
        cv = _to_float(candidate.get(key))
        rv = _to_float(reference.get(key))
        if cv is not None and rv is not None and abs(cv - rv) < 1e-9:
            score += 1
    return score


def _best_reference(row: Dict, references: list) -> Optional[Dict]:
    scored = [(_score_match(row, ref), ref) for ref in references]
    scored = [x for x in scored if x[0] > 0]
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def check_row(
    row: Dict,
    *,
    references: Optional[list] = None,
    abs_tolerance: float = 0.10,
    rel_tolerance: float = 0.25,
) -> Dict:
    """Compare one result row to the closest available GPU reference row."""
    references = references if references is not None else _load_reference_rows()
    ref = _best_reference(row, references)
    checks = []
    if ref is None:
        return {
            "status": "no_reference",
            "checks": checks,
            "message": "No compatible Llama GPU reference row found",
        }

    for metric, aliases in METRIC_ALIASES.items():
        observed = _to_float(row.get(metric))
        expected = None
        expected_key = None
        for alias in aliases:
            expected = _to_float(ref.get(alias))
            if expected is not None:
                expected_key = alias
                break
        if observed is None or expected is None:
            continue
        delta = abs(observed - expected)
        limit = max(abs_tolerance, rel_tolerance * max(abs(expected), 1e-6))
        checks.append({
            "metric": metric,
            "reference_metric": expected_key,
            "observed": observed,
            "expected": expected,
            "delta": delta,
            "limit": limit,
            "pass": delta <= limit,
        })

    status = "pass" if checks and all(c["pass"] for c in checks) else "fail"
    if not checks:
        status = "no_comparable_metrics"
    return {"status": status, "checks": checks, "reference": ref}


def check_csv(
    csv_path: str,
    *,
    output_path: Optional[str] = None,
    abs_tolerance: float = 0.10,
    rel_tolerance: float = 0.25,
) -> Dict:
    references = _load_reference_rows()
    rows = list(_load_csv_rows(csv_path))
    results = [
        check_row(
            row,
            references=references,
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
        )
        for row in rows
    ]
    summary = {
        "csv_path": csv_path,
        "n_rows": len(rows),
        "n_fail": sum(r["status"] == "fail" for r in results),
        "results": results,
    }
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--abs_tolerance", type=float, default=0.10)
    parser.add_argument("--rel_tolerance", type=float, default=0.25)
    args = parser.parse_args()
    summary = check_csv(
        args.csv,
        output_path=args.output,
        abs_tolerance=args.abs_tolerance,
        rel_tolerance=args.rel_tolerance,
    )
    print(json.dumps({
        "status": "fail" if summary["n_fail"] else "pass",
        "n_rows": summary["n_rows"],
        "n_fail": summary["n_fail"],
    }, indent=2))
    raise SystemExit(1 if summary["n_fail"] else 0)


if __name__ == "__main__":
    main()
