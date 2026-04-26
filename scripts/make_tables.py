"""Pivot APRS result CSVs into paper-ready LaTeX and Markdown tables.

The input is the CSV schema emitted by run_obfuscation_pipeline.py and by
scripts/run_final_eval.sh's aggregator.  The renderer is intentionally tolerant
of older CSVs: missing optional columns render as em dashes, and rows with no
model name are skipped instead of producing blank model sections.
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Iterable, Sequence


# Each column is (candidate CSV keys, display label).  The first present,
# non-empty value is used.  This lets tables work with both old and new result
# files and with undefended rows that store baseline metrics in *_undefended.
SECURITY_COLS = [
    (("arditi_score_defended", "arditi_score_undefended"), "Arditi"),
    (("pca8_score_defended", "pca8_score_undefended"), "PCA-8"),
    (("perlayer_score_defended", "perlayer_score_undefended"), "Per-layer"),
    (("leace_score_defended", "leace_score_undefended"), "LEACE"),
    (("harmbench_asr_post_gcg", "gcg_asr"), "GCG"),
    (("harmbench_asr_post_autodan", "autodan_asr"), "AutoDAN"),
    (("harmbench_asr_post_pair", "pair_asr"), "PAIR"),
    (("harmbench_asr_post_renellm", "renellm_asr"), "ReNeLLM"),
    (("softopt_asr",), "SoftOpt"),
]

UTILITY_COLS = [
    (("pile_bpb",), "Pile BPB"),
    (("alpaca_bpb",), "Alpaca BPB"),
    (("gsm8k_exact_match", "gsm8k_exact_match_undefended"), "GSM8K"),
    (("math500_exact_match", "math500_exact_match_undefended"), "MATH500"),
    (("mmlu_acc", "mmlu_acc_undefended"), "MMLU"),
    (("alpacaeval_lc_win_rate",), "AlpacaEval LC-WR"),
    (("xstest_over_refusal_rate",), "XSTest-OR"),
]


def _float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _first_value(row: dict, keys: Sequence[str]):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _fmt(row: dict, keys: Sequence[str]) -> str:
    value = _first_value(row, keys)
    v = _float(value)
    return "--" if v is None else f"{v:.3f}"


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_writer_only(row: dict) -> bool:
    return _truthy(row.get("writer_only")) or "writer_only" in row.get("run_tag", "")


ROW_ORDER = [
    ("Undefended", lambda r: r.get("defense_type") == "none"),
    (
        "APRS full eps=0.025",
        lambda r: (
            r.get("defense_type") == "obfuscation"
            and r.get("projection_mode") == "full"
            and _float(r.get("epsilon")) == 0.025
            and not _is_writer_only(r)
        ),
    ),
    (
        "APRS hadamard eps=0.3",
        lambda r: (
            r.get("defense_type") == "obfuscation"
            and r.get("projection_mode") == "hadamard"
            and _float(r.get("epsilon")) == 0.3
        ),
    ),
    (
        "APRS scalar eps=0.3",
        lambda r: (
            r.get("defense_type") == "obfuscation"
            and r.get("projection_mode") == "scalar_projection"
            and _float(r.get("epsilon")) == 0.3
        ),
    ),
    (
        "APRS full eps=0.025 writer-only",
        lambda r: (
            r.get("defense_type") == "obfuscation"
            and r.get("projection_mode") == "full"
            and _float(r.get("epsilon")) == 0.025
            and _is_writer_only(r)
        ),
    ),
    ("Surgical", lambda r: r.get("defense_type") == "surgical"),
    ("CAST", lambda r: r.get("defense_type") == "cast"),
    ("Circuit Breakers", lambda r: r.get("defense_type") == "circuit_breakers"),
    ("AlphaSteer", lambda r: r.get("defense_type") == "alphasteer"),
]


def _model_sort_key(item):
    model, _ = item
    return model.split("/")[-1].lower()


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def build_table(rows_by_model, cols, caption, label):
    """Render LaTeX and Markdown tables, one table per model."""
    tex_chunks, md_chunks = [], []
    for model, rows in sorted(rows_by_model.items(), key=_model_sort_key):
        pretty_model = model.split("/")[-1]
        col_spec = "l" + "c" * len(cols)
        header = " & ".join(["Defense"] + [_latex_escape(c[1]) for c in cols]) + r" \\"
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{_latex_escape(caption)} --- {_latex_escape(pretty_model)}}}",
            rf"\label{{{label}-{_latex_escape(pretty_model)}}}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header,
            r"\midrule",
        ]
        md = [
            f"### {caption} - {pretty_model}",
            "",
            "| Defense | " + " | ".join(c[1] for c in cols) + " |",
            "| --- | " + " | ".join("---" for _ in cols) + " |",
        ]

        for name, matcher in ROW_ORDER:
            match = [r for r in rows if matcher(r)]
            if not match:
                continue
            row = match[-1]
            vals = [_fmt(row, c[0]) for c in cols]
            lines.append(" & ".join([_latex_escape(name)] + vals) + r" \\")
            md.append(f"| {name} | " + " | ".join(vals) + " |")

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
        tex_chunks.append("\n".join(lines))
        md_chunks.append("\n".join(md))

    return "\n".join(tex_chunks), "\n\n".join(md_chunks)


def _load_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("model")]


def _warn_missing_columns(rows: Iterable[dict], cols) -> None:
    available = set()
    for row in rows:
        available.update(row.keys())
    missing_groups = [
        keys for keys, _ in cols
        if not any(key in available for key in keys)
    ]
    if missing_groups:
        printable = ["/".join(keys) for keys in missing_groups]
        print(f"[WARN] Missing optional metric columns: {', '.join(printable)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="all_results.csv")
    ap.add_argument("--out_dir", default="results/tables")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = _load_rows(args.csv)

    _warn_missing_columns(rows, SECURITY_COLS + UTILITY_COLS)

    rows_by_model = defaultdict(list)
    for row in rows:
        rows_by_model[row["model"]].append(row)

    security_tex, security_md = build_table(
        rows_by_model,
        SECURITY_COLS,
        "Main Security Metrics",
        "tab:security",
    )
    utility_tex, utility_md = build_table(
        rows_by_model,
        UTILITY_COLS,
        "Main Utility Metrics",
        "tab:utility",
    )

    for name, content in [
        ("security_main.tex", security_tex),
        ("security_main.md", security_md),
        ("utility.tex", utility_tex),
        ("utility.md", utility_md),
        # Backwards-compatible filenames used by earlier drafts.
        ("asr_main.tex", security_tex),
        ("asr_main.md", security_md),
    ]:
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as f:
            f.write(content)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
