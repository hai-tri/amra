# AMRA: Abliteration Mitigation via Refusal Aliases

Code for the paper *Abliteration Mitigation via Refusal Aliases*.

A post-hoc weight-editing defense that obscures the refusal direction to resist abliteration. Rank-_k_ updates remap refusal-inducing activations to random aliases, with SVD-based corrections to downstream readers preserving model behavior. Improves post-abliteration refusal by 2.16 pts on Llama-3-8B (<0.5pp MMLU loss) and 14.70 pts on Gemma-2-9B.

## Setup

```bash
git clone --recurse-submodules <repo> && cd <repo>
uv venv --python 3.12 && source .venv/bin/activate     # or: python3.12 -m venv
uv sync                                                 # or: pip install -r requirements.txt
huggingface-cli login                                   # for gated Llama-3, Gemma-2
```
