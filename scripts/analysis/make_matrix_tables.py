"""Emit reader_matrices.csv and writer_matrices.csv for the paper.

Names are pulled from ModelComponents so the tables stay in sync with code.
Qwen attn readers are unimplemented; left blank intentionally.
"""

import csv
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from obfuscation_utils import ModelComponents


def _fake_llama():
    m = MagicMock()
    layer = SimpleNamespace(
        input_layernorm=nn.Identity(),
        post_attention_layernorm=nn.Identity(),
        self_attn=SimpleNamespace(
            q_proj=nn.Linear(1, 1), k_proj=nn.Linear(1, 1),
            v_proj=nn.Linear(1, 1), o_proj=nn.Linear(1, 1),
        ),
        mlp=SimpleNamespace(
            gate_proj=nn.Linear(1, 1), up_proj=nn.Linear(1, 1),
            down_proj=nn.Linear(1, 1),
        ),
    )
    m.model = SimpleNamespace(
        layers=[layer], norm=nn.Identity(),
    )
    m.lm_head = nn.Identity()
    return m


def _fake_qwen():
    m = MagicMock()
    layer = SimpleNamespace(
        ln_1=nn.Identity(), ln_2=nn.Identity(),
        attn=SimpleNamespace(c_proj=nn.Linear(1, 1)),
        mlp=SimpleNamespace(
            w1=nn.Linear(1, 1), w2=nn.Linear(1, 1), c_proj=nn.Linear(1, 1),
        ),
    )
    m.transformer = SimpleNamespace(h=[layer], ln_f=nn.Identity())
    m.lm_head = nn.Identity()
    if hasattr(m, "model"):
        del m.model
    return m


FAMILIES = [
    ("llama",   _fake_llama),
    ("gemma",   _fake_llama),
    ("mistral", _fake_llama),
    ("qwen",    _fake_qwen),
]


def _names(projs):
    try:
        return [n for n, _ in projs]
    except NotImplementedError:
        return []


def main():
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)

    rows_r, rows_w = [], []
    max_attn_r = max_mlp_r = 0
    for fam, factory in FAMILIES:
        c = ModelComponents(factory())
        try:
            attn_r = _names(c.get_attn_reader_projs(0))
        except NotImplementedError:
            attn_r = []
        mlp_r = _names(c.get_mlp_reader_projs(0))
        attn_w = type(c.get_attn_output_proj(0)).__name__ if False else None
        # Names for writers are fixed per arch; hard-code from code paths.
        attn_w = "o_proj" if c.arch == "llama" else "c_proj"
        mlp_w  = "down_proj" if c.arch == "llama" else "c_proj"
        rows_r.append((fam, attn_r, mlp_r))
        rows_w.append((fam, attn_w, mlp_w))
        max_attn_r = max(max_attn_r, len(attn_r))
        max_mlp_r  = max(max_mlp_r, len(mlp_r))

    reader_path = os.path.join(out_dir, "reader_matrices.csv")
    with open(reader_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["model_family"] \
            + [f"attn_reader_{i+1}" for i in range(max_attn_r)] \
            + [f"mlp_reader_{i+1}"  for i in range(max_mlp_r)]
        w.writerow(header)
        for fam, attn_r, mlp_r in rows_r:
            w.writerow([fam]
                       + attn_r + [""] * (max_attn_r - len(attn_r))
                       + mlp_r  + [""] * (max_mlp_r  - len(mlp_r)))
    print(f"wrote {reader_path}")

    writer_path = os.path.join(out_dir, "writer_matrices.csv")
    with open(writer_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_family", "attn_writer", "mlp_writer"])
        for row in rows_w:
            w.writerow(row)
    print(f"wrote {writer_path}")


if __name__ == "__main__":
    main()
