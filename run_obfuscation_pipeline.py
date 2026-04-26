"""
Full pipeline for the Representational Obfuscation defense.

Usage::

    python run_obfuscation_pipeline.py --model_path meta-llama/Meta-Llama-3-8B-Instruct

Pipeline stages:
    1. generate_directions    — extract candidate refusal directions   (existing)
    2. select_direction       — pick best r̂                            (existing)
    2b. pre-defense integrity — collect residual stats & logits        (NEW)
    3. apply_obfuscation      — apply the defense                      (NEW)
    3b. post-defense integrity— variance diffs & output KL             (NEW)
    4. completions            — evaluate refusal on harmful/harmless    (existing)
    5. loss_evals             — evaluate utility                       (existing)
    5b. llamaguard            — Llama Guard safety classification        (NEW)
    6. evaluate_abliteration  — standard abliteration on defended model (NEW)
    7. evaluate_adaptive      — PCA / per-layer / sublayer attacks      (NEW)
"""

import argparse
import functools
import json
import os
import random
import sys
import torch

# Add the existing refusal_direction package to the path
_REFUSAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)

from dataset.load_dataset import load_dataset
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.utils.hook_utils import (
    get_activation_addition_input_pre_hook,
    get_all_direction_ablation_hooks,
)

from obfuscation_config import ObfuscationConfig
from obfuscation_utils import (
    ModelComponents,
    collect_writer_output_refusal_directions,
    writer_output_direction_cosine_summary,
)
from device_utils import empty_cache as _dev_empty_cache
from apply_obfuscation import apply_obfuscation
from defenses.apply_surgical import apply_surgical
from defenses.apply_cast import apply_cast
from defenses.apply_circuit_breakers import apply_circuit_breakers
from attacks.evaluate_abliteration import evaluate_abliteration_resistance
from attacks.evaluate_adaptive_attack import run_all_adaptive_attacks
from evaluations.evaluate_integrity import (
    collect_pre_defense_measurements,
    evaluate_defense_integrity,
)
from attacks.evaluate_leace_attack import leace_attack


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the full obfuscation-defense pipeline."
    )
    parser.add_argument("--model_path", type=str, required=True)

    # Defense selection
    parser.add_argument("--defense_type", type=str, default="obfuscation",
                        choices=["obfuscation", "surgical", "cast", "circuit_breakers", "alphasteer"],
                        help="Defense mechanism to evaluate (default: obfuscation)")
    # Surgical hyperparameters
    parser.add_argument("--surgical_ablation_coeff", type=float, default=1.0,
                        help="Surgical ablation coefficient (default: 1.0)")
    parser.add_argument("--surgical_actadd_coeff", type=float, default=0.0,
                        help="Surgical activation addition coefficient (default: 0.0)")
    # CAST hyperparameters
    parser.add_argument("--cast_strength", type=float, default=1.5,
                        help="CAST behavior vector strength (default: 1.5)")
    parser.add_argument("--cast_threshold", type=float, default=0.02,
                        help="CAST condition cosine similarity threshold (default: 0.02)")
    # Circuit Breakers hyperparameters
    parser.add_argument("--cb_steps", type=int, default=150,
                        help="Circuit Breakers training steps (default: 150)")
    parser.add_argument("--cb_lr", type=float, default=1e-4,
                        help="Circuit Breakers learning rate (default: 1e-4)")
    parser.add_argument("--cb_lora_rank", type=int, default=16,
                        help="Circuit Breakers LoRA rank (default: 16)")
    parser.add_argument("--cb_batch_size", type=int, default=4,
                        help="Circuit Breakers training batch size (default: 4)")
    parser.add_argument("--cb_coeff_max", type=float, default=1.0,
                        help="Circuit Breakers peak RR loss coefficient (default: 1.0)")
    parser.add_argument("--cb_retain_coeff_max", type=float, default=1.0,
                        help="Circuit Breakers peak retention loss coefficient (default: 1.0)")
    # AlphaSteer hyperparameters
    parser.add_argument("--alphasteer_strength", type=float, default=0.4,
                        help="AlphaSteer steering strength (default: 0.4)")
    parser.add_argument("--alphasteer_null_ratio", type=float, default=0.5,
                        help="AlphaSteer null-space ratio (default: 0.5)")
    parser.add_argument("--alphasteer_lambda", type=float, default=10.0,
                        help="AlphaSteer regularisation coefficient (default: 10.0)")

    # Obfuscation hyperparameters
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Std of random alias vectors (default: 0.1)")
    parser.add_argument("--num_pertinent_layers", type=int, default=None,
                        help="Override number of pertinent layers (default: auto-detect from data)")
    parser.add_argument("--num_calibration_prompts", type=int, default=32,
                        help="Number of prompts for calibration (default: 32)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_writers", type=str, default="both",
                        choices=["both", "attn_only", "mlp_only"])
    parser.add_argument("--no_separate_aliases", action="store_true",
                        help="Use the same alias for W_O and W_down")
    parser.add_argument("--pca_top_k", type=int, default=8,
                        help="Number of PCA directions for the adaptive attack")
    parser.add_argument("--pertinent_layers", type=str, default=None,
                        help="Comma-separated list of specific layers to patch, e.g. '9,10'")
    parser.add_argument("--projection_mode", type=str, default="hadamard",
                        choices=["hadamard", "binary", "mask", "scalar_projection", "full"],
                        help="Projection mode: hadamard (default), binary, mask, scalar_projection, or full")
    parser.add_argument("--per_layer_direction", action="store_true",
                        help="Use per-layer refusal directions instead of global r̂")
    parser.add_argument("--writer_output_directions", action="store_true",
                        help=("Use per-writer output refusal directions: "
                              "W_O^l gets its own attention-output direction "
                              "and W_down^l gets its own MLP-output direction"))
    parser.add_argument("--obfuscation_writer_only", action="store_true",
                        help="Skip reader (Q/K/V/gate/up) patches; apply writer (o_proj/down_proj) edits only. Ablation.")

    # Pipeline control
    parser.add_argument("--skip_direction_extraction", action="store_true",
                        help="Skip direction extraction if artifacts already exist")
    parser.add_argument("--harmful_only_calibration", action="store_true",
                        help="Calibrate rank-one patches on harmful prompts only (ablation baseline)")
    parser.add_argument("--harmless_ratio", type=float, default=0.5,
                        help="Fraction of calibration prompts drawn from harmless set (default: 0.5)")
    parser.add_argument("--skip_evaluations", action="store_true",
                        help="Skip completion/loss evaluations")
    parser.add_argument("--undefended_only", action="store_true",
                        help="Run only undefended baseline evaluations (no defense applied)")
    parser.add_argument("--llamaguard", action="store_true",
                        help="Run Llama Guard evaluation on completions (Stage 5b)")
    parser.add_argument("--llamaguard_model", type=str, default="meta-llama/Llama-Guard-3-8B",
                        help="Llama Guard model ID (default: meta-llama/Llama-Guard-3-8B)")
    parser.add_argument("--softopt", action="store_true",
                        help="Run SoftOpt white-box attack evaluation (Stage 7b)")
    parser.add_argument("--softopt_steps", type=int, default=500,
                        help="SoftOpt optimization steps (default: 500)")
    parser.add_argument("--softopt_limit", type=int, default=None,
                        help="Limit number of SoftOpt behaviors to test")
    parser.add_argument("--softopt_benchmark", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data", "harmbench_test_std.json"),
                        help="Path to HarmBench-format benchmark JSON")
    parser.add_argument("--skip_harmbench", action="store_true",
                        help="Skip HarmBench ASR evaluation (Stage 4b)")
    parser.add_argument("--skip_xstest", action="store_true",
                        help="Skip XSTest over-refusal evaluation (Stage 4c)")
    parser.add_argument("--skip_lm_harness", action="store_true",
                        help="Skip GSM8k/MATH500/MMLU evaluation (Stage 9)")
    parser.add_argument("--ce_loss_batch_size", type=int, default=4,
                        help="Batch size for Pile/Alpaca CE-loss evaluation (default: 4)")
    parser.add_argument("--ce_loss_n_batches", type=int, default=64,
                        help=("Number of Pile/Alpaca CE-loss batches "
                              "(default: 64, i.e. 256 sequences at batch size 4)"))
    parser.add_argument("--lm_harness_tasks", type=str, default="gsm8k,math500,mmlu",
                        help="Comma-separated lm-harness tasks to run (default: gsm8k,math500,mmlu)")
    parser.add_argument("--lm_harness_n", type=int, default=100,
                        help="Number of examples to sample per task (default: 100)")
    parser.add_argument("--skip_alpacaeval", action="store_true",
                        help="Skip AlpacaEval generation-quality evaluation (Stage 9b)")
    parser.add_argument("--alpacaeval_n", type=int, default=100,
                        help="Number of AlpacaEval prompts to sample (default: 100, max 805). "
                             "Use 805 for the full benchmark.")
    parser.add_argument("--alpacaeval_max_new_tokens", type=int, default=512,
                        help="Max new tokens per AlpacaEval response (default: 512)")
    parser.add_argument("--alpacaeval_skip_judge", action="store_true",
                        help="Generate AlpacaEval completions but skip the "
                             "LLM-judge win-rate step (saves completions only)")
    parser.add_argument("--alpacaeval_annotator", type=str,
                        default="alpaca_eval_gpt4_turbo_fn",
                        help="alpaca_eval annotator config "
                             "(default: alpaca_eval_gpt4_turbo_fn; requires OPENAI_API_KEY)")
    parser.add_argument("--harmbench_csv", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data", "harmbench_behaviors_text_test.csv"),
                        help="Path to HarmBench behaviors CSV")
    parser.add_argument("--harmbench_n", type=int, default=100,
                        help="Number of HarmBench behaviors to sample (default: 100)")
    parser.add_argument("--skip_leace", action="store_true",
                        help="Skip LEACE attack evaluation (Stage 7c)")
    parser.add_argument("--skip_heretic", action="store_true",
                        help="Skip Heretic attack evaluation (Stage 8)")
    parser.add_argument("--heretic_trials", type=int, default=50,
                        help="Number of Heretic Optuna trials (default: 50)")
    # GCG
    parser.add_argument("--gcg", action="store_true",
                        help="Run GCG white-box attack (Zou et al. 2023)")
    parser.add_argument("--gcg_steps", type=int, default=500,
                        help="GCG optimisation steps per behavior (default: 500)")
    parser.add_argument("--gcg_suffix_len", type=int, default=20,
                        help="GCG adversarial suffix length in tokens (default: 20)")
    parser.add_argument("--gcg_n_behaviors", type=int, default=25,
                        help="Number of behaviors to attack with GCG (default: 25)")
    parser.add_argument("--gcg_topk", type=int, default=256,
                        help="GCG candidate pool size per position (default: 256)")
    parser.add_argument("--gcg_batch_size", type=int, default=128,
                        help="GCG candidate batch size per step (default: 128)")
    # AutoDAN
    parser.add_argument("--autodan", action="store_true",
                        help="Run AutoDAN-GA attack (Liu et al. 2023)")
    parser.add_argument("--autodan_steps", type=int, default=100,
                        help="AutoDAN GA generations (default: 100)")
    parser.add_argument("--autodan_population", type=int, default=64,
                        help="AutoDAN population size (default: 64)")
    parser.add_argument("--autodan_n_behaviors", type=int, default=25,
                        help="Number of behaviors to attack with AutoDAN (default: 25)")
    # CipherChat
    parser.add_argument("--cipherchat", action="store_true",
                        help="Run CipherChat black-box attack (Yuan et al. 2023)")
    parser.add_argument("--cipherchat_ciphers", type=str, default="caesar,base64",
                        help="Comma-separated cipher types to try: "
                             "caesar,morse,ascii,base64 (default: caesar,base64)")
    # PAIR
    parser.add_argument("--pair", action="store_true",
                        help="Run PAIR black-box attack (Chao et al. 2023)")
    parser.add_argument("--pair_streams", type=int, default=5,
                        help="PAIR parallel streams per behavior (default: 5)")
    parser.add_argument("--pair_iterations", type=int, default=5,
                        help="PAIR refinement iterations per stream (default: 5)")
    parser.add_argument("--pair_n_behaviors", type=int, default=25,
                        help="Number of behaviors to attack with PAIR (default: 25)")
    parser.add_argument("--pair_attacker", type=str, default=None,
                        help="HF path for a separate attacker LLM; "
                             "defaults to self-PAIR (uses target model)")
    # ReNeLLM
    parser.add_argument("--renellm", action="store_true",
                        help="Run ReNeLLM nested jailbreak attack (Ding et al. 2023)")
    parser.add_argument("--renellm_strategies", type=int, default=2,
                        help="Number of rewriting strategies to chain (default: 2)")
    parser.add_argument("--renellm_attempts", type=int, default=3,
                        help="Scenario attempts per behavior (default: 3)")
    parser.add_argument("--artifact_subdir", type=str, default="obfuscation",
                        help=("Subdirectory under the model run artifact dir "
                              "for defense/eval artifacts (default: obfuscation)"))
    parser.add_argument("--save_csv", type=str, default=None,
                        help="Path to CSV file to append results to (e.g. results.csv)")
    return parser.parse_args()


SYSTEM_PROMPT = "You are a helpful assistant."


def _generate_harmbench_for_attack(
    *,
    label: str,
    model,
    tokenizer,
    tokenize_fn,
    behaviors,
    attacked_prompts=None,
    responses=None,
    fwd_pre_hooks=None,
    fwd_hooks=None,
    max_new_tokens=256,
    classifier_model_id="cais/HarmBench-Llama-2-13b-cls",
    artifact_dir=None,
):
    from evaluations.evaluate_harmbench import (
        generate_responses_for_prompts,
        score_harmbench_responses,
    )

    if not behaviors:
        return None

    if responses is None:
        if attacked_prompts is None:
            raise ValueError(f"{label}: expected attacked prompts or responses")
        responses = generate_responses_for_prompts(
            model=model,
            tokenizer=tokenizer,
            tokenize_fn=tokenize_fn,
            prompts=attacked_prompts,
            fwd_pre_hooks=fwd_pre_hooks or [],
            fwd_hooks=fwd_hooks or [],
            max_new_tokens=max_new_tokens,
            batch_size=4,
        )

    metadata = None
    if attacked_prompts is not None:
        metadata = [{"attacked_prompt": prompt} for prompt in attacked_prompts]

    artifact_path = None
    if artifact_dir is not None:
        artifact_path = os.path.join(artifact_dir, f"harmbench_post_{label}.json")

    _device = next(model.parameters()).device
    model.to("cpu")
    _dev_empty_cache()
    try:
        return score_harmbench_responses(
            prompts=behaviors,
            responses=responses,
            classifier_model_id=classifier_model_id,
            artifact_path=artifact_path,
            metadata=metadata,
        )
    finally:
        model.to(_device)


def load_mlabonne_datasets(n_train=128, n_val=32, seed=42):
    """
    Load harmful/harmless datasets from mlabonne's HuggingFace repos
    (same sources used by Heretic).

      - harmful:  mlabonne/harmful_behaviors
      - harmless: mlabonne/harmless_alpaca
    """
    from datasets import load_dataset as hf_load_dataset

    print("[data] Loading mlabonne/harmful_behaviors …")
    harmful_ds = hf_load_dataset("mlabonne/harmful_behaviors", split="train")
    # Both mlabonne datasets use a 'text' column (confirmed from Heretic config)
    col = "text" if "text" in harmful_ds.column_names else "instruction"
    harmful_all = [row[col] for row in harmful_ds]

    print("[data] Loading mlabonne/harmless_alpaca …")
    harmless_ds = hf_load_dataset("mlabonne/harmless_alpaca", split="train")
    col = "text" if "text" in harmless_ds.column_names else "instruction"
    harmless_all = [row[col] for row in harmless_ds]

    random.seed(seed)
    random.shuffle(harmful_all)
    random.shuffle(harmless_all)

    harmful_train = harmful_all[:n_train]
    harmful_val = harmful_all[n_train : n_train + n_val]
    harmless_train = harmless_all[:n_train]
    harmless_val = harmless_all[n_train : n_train + n_val]

    print(f"[data] harmful  — train: {len(harmful_train)}, val: {len(harmful_val)} "
          f"(total available: {len(harmful_all)})")
    print(f"[data] harmless — train: {len(harmless_train)}, val: {len(harmless_val)} "
          f"(total available: {len(harmless_all)})")

    return harmful_train, harmless_train, harmful_val, harmless_val


def filter_data(model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """Keep only harmful prompts the model actually refuses and harmless prompts it doesn't."""
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    harmful_train_scores = get_refusal_scores(
        model_base.model, harmful_train,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
    )
    harmless_train_scores = get_refusal_scores(
        model_base.model, harmless_train,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
    )
    harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
    harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    harmful_val_scores = get_refusal_scores(
        model_base.model, harmful_val,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
    )
    harmless_val_scores = get_refusal_scores(
        model_base.model, harmless_val,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
    )
    harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
    harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)

    return harmful_train, harmless_train, harmful_val, harmless_val


def run_pipeline(args):
    model_alias = os.path.basename(args.model_path)
    cfg = Config(model_alias=model_alias, model_path=args.model_path)

    obf_cfg = ObfuscationConfig(
        epsilon=args.epsilon,
        num_pertinent_layers=args.num_pertinent_layers,
        num_calibration_prompts=args.num_calibration_prompts,
        separate_attn_mlp_aliases=not args.no_separate_aliases,
        seed=args.seed,
        patch_writers=args.patch_writers,
        projection_mode=args.projection_mode,
        per_layer_direction=args.per_layer_direction,
        writer_output_directions=args.writer_output_directions,
    )

    # Match Heretic's dataset sizes: 400 for direction extraction, 100 for val
    cfg.n_train = 400
    cfg.n_val = 100

    cfg.ce_loss_batch_size = args.ce_loss_batch_size
    cfg.ce_loss_n_batches = args.ce_loss_n_batches

    artifact_dir = cfg.artifact_path()
    artifact_subdir = args.artifact_subdir
    if args.undefended_only and artifact_subdir == "obfuscation":
        artifact_subdir = "undefended"
    obf_artifact_dir = os.path.join(artifact_dir, artifact_subdir)
    os.makedirs(obf_artifact_dir, exist_ok=True)

    # Save obfuscation config
    with open(os.path.join(obf_artifact_dir, "obfuscation_config.json"), "w") as f:
        json.dump(obf_cfg.__dict__, f, indent=4)

    # ==================================================================
    # Load model and datasets
    # ==================================================================
    print("=" * 60)
    print("Loading model and datasets …")
    print("=" * 60)

    model_base = construct_model_base(cfg.model_path)

    # Inject system prompt into the tokenizer (Gemma models don't support system prompts)
    _is_gemma = "gemma" in cfg.model_path.lower()
    if not _is_gemma:
        model_base.tokenize_instructions_fn = functools.partial(
            model_base.tokenize_instructions_fn, system=SYSTEM_PROMPT,
        )
        print(f"[config] System prompt: \"{SYSTEM_PROMPT}\"")
    else:
        print("[config] System prompt: (none — Gemma does not support system prompts)")

    # On TPU/XLA, patch model.generate so every call uses bucket-padded inputs
    # and a static KV cache — amortizes graph compilation across attack/eval
    # suites. Semantics (greedy / sampled tokens) are unchanged.
    try:
        from scripts.tpu_utils import is_xla_env, patch_model_for_xla
        if is_xla_env():
            patch_model_for_xla(model_base.model, model_base.tokenizer)
    except Exception as _e:
        print(f"[xla-generate] skipped patch: {_e}")

    # Load mlabonne datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_mlabonne_datasets(
        n_train=cfg.n_train, n_val=cfg.n_val,
    )
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        model_base, harmful_train, harmless_train, harmful_val, harmless_val,
    )

    # ==================================================================
    # Stage 1–2: Generate / select refusal direction (existing pipeline)
    # ==================================================================
    gen_dir = os.path.join(artifact_dir, "generate_directions")
    mean_diffs_path = os.path.join(gen_dir, "mean_diffs.pt")
    direction_path = os.path.join(artifact_dir, "direction.pt")

    _meta_path = os.path.join(artifact_dir, "direction_metadata.json")
    if (args.skip_direction_extraction
            and os.path.exists(direction_path)
            and os.path.exists(mean_diffs_path)
            and os.path.exists(_meta_path)):
        print("Loading cached direction artifacts …")
        mean_diffs = torch.load(mean_diffs_path, map_location="cpu")
        direction = torch.load(direction_path, map_location="cpu")
        with open(os.path.join(artifact_dir, "direction_metadata.json")) as f:
            meta = json.load(f)
        pos, layer = meta["pos"], meta["layer"]
    else:
        print("=" * 60)
        print("Stage 1: Generating candidate refusal directions …")
        print("=" * 60)
        os.makedirs(gen_dir, exist_ok=True)
        mean_diffs = generate_directions(
            model_base, harmful_train, harmless_train,
            artifact_dir=gen_dir,
        )
        torch.save(mean_diffs, mean_diffs_path)

        print("=" * 60)
        print("Stage 2: Selecting best refusal direction …")
        print("=" * 60)
        sel_dir = os.path.join(artifact_dir, "select_direction")
        os.makedirs(sel_dir, exist_ok=True)
        pos, layer, direction = select_direction(
            model_base, harmful_val, harmless_val,
            mean_diffs, artifact_dir=sel_dir,
        )
        with open(os.path.join(artifact_dir, "direction_metadata.json"), "w") as f:
            json.dump({"pos": pos, "layer": layer}, f, indent=4)
        torch.save(direction, direction_path)

    # Load ablation scores produced by select_direction — used for causal layer selection
    ablation_scores_path = os.path.join(artifact_dir, "select_direction", "direction_evaluations.json")
    ablation_scores = None
    if os.path.exists(ablation_scores_path):
        with open(ablation_scores_path) as f:
            ablation_scores = json.load(f)
        print(f"Loaded {len(ablation_scores)} ablation score entries for causal layer selection.")

    # Keep a copy of the original direction for diagnostics
    original_direction = direction.clone()

    print(f"\nRefusal direction: pos={pos}, layer={layer}, "
          f"||r||={direction.norm().item():.4f}")

    # ==================================================================
    # Stage 2b: Pre-defense integrity measurements  (NEW)
    # ==================================================================
    print("=" * 60)
    print("Stage 2b: Collecting pre-defense integrity measurements …")
    print("=" * 60)

    components = ModelComponents(model_base.model)
    pre_measurements = collect_pre_defense_measurements(
        model=model_base.model,
        components=components,
        harmful_prompts=harmful_val,
        harmless_prompts=harmless_val,
        tokenize_fn=model_base.tokenize_instructions_fn,
        num_prompts=min(obf_cfg.num_calibration_prompts, len(harmful_val)),
    )

    # Measure undefended refusal score (before defense is applied)
    print("[pre-defense] Measuring undefended refusal score …")
    undefended_refusal_scores = get_refusal_scores(
        model_base.model, harmful_val,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
    )
    undefended_refusal_mean = undefended_refusal_scores.mean().item()
    print(f"[pre-defense] Undefended refusal score: {undefended_refusal_mean:.4f}")

    writer_output_dirs_pre = None
    if args.defense_type == "obfuscation" and not args.undefended_only:
        print("[pre-defense] Collecting writer-output refusal directions …")
        writer_output_dirs_pre = collect_writer_output_refusal_directions(
            model=model_base.model,
            components=components,
            harmful_prompts=harmful_val,
            harmless_prompts=harmless_val,
            tokenize_fn=model_base.tokenize_instructions_fn,
            num_prompts=min(obf_cfg.num_calibration_prompts,
                            len(harmful_val), len(harmless_val)),
        )

    # Run attacks on the undefended model as baselines
    print("[pre-defense] Running attacks on undefended model …")
    undefended_abl = evaluate_abliteration_resistance(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        tokenize_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        attn_modules=model_base.model_attn_modules,
        mlp_modules=model_base.model_mlp_modules,
        harmful_prompts=harmful_val,
        benign_prompts=harmless_val,
        original_direction=original_direction,
        refusal_toks=model_base.refusal_toks,
    )
    undefended_adaptive = run_all_adaptive_attacks(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        tokenize_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        attn_modules=model_base.model_attn_modules,
        mlp_modules=model_base.model_mlp_modules,
        harmful_prompts=harmful_val,
        benign_prompts=harmless_val,
        original_direction=original_direction,
        refusal_toks=model_base.refusal_toks,
        pca_top_k=args.pca_top_k,
    )
    print(f"[pre-defense] Undefended post-abliteration: "
          f"{undefended_abl['arditi_refusal_score']:.4f}")
    print(f"[pre-defense] Undefended PCA-{args.pca_top_k}: "
          f"{undefended_adaptive['pca']['post_attack_refusal_score']:.4f}")
    print(f"[pre-defense] Undefended per-layer: "
          f"{undefended_adaptive['per_layer']['post_attack_refusal_score']:.4f}")

    undefended_leace = None
    if not args.skip_leace:
        print("[pre-defense] Running LEACE on undefended model …")
        undefended_leace = leace_attack(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            attn_modules=model_base.model_attn_modules,
            mlp_modules=model_base.model_mlp_modules,
            harmful_prompts=harmful_val,
            benign_prompts=harmless_val,
            original_direction=original_direction,
            refusal_toks=model_base.refusal_toks,
        )
        print(f"[pre-defense] Undefended LEACE: "
              f"{undefended_leace['post_attack_refusal_score']:.4f}")

    if args.undefended_only:
        print("\n" + "=" * 60)
        print("--undefended_only: Running black-box attacks on undefended model …")
        print("=" * 60)

        _undef_artifact_dir = obf_artifact_dir
        os.makedirs(_undef_artifact_dir, exist_ok=True)

        _undef_gcg = _undef_autodan = _undef_cipher = None
        _undef_pair = _undef_renellm = _undef_softopt = None

        if args.gcg:
            from attacks.evaluate_gcg import evaluate_gcg
            print("Stage U-1: GCG on undefended model …")
            _undef_gcg = evaluate_gcg(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                tokenize_fn=model_base.tokenize_instructions_fn,
                harmful_prompts=harmful_val,
                refusal_toks=model_base.refusal_toks,
                suffix_len=args.gcg_suffix_len,
                num_steps=args.gcg_steps,
                topk=args.gcg_topk,
                batch_size=args.gcg_batch_size,
                n_behaviors=args.gcg_n_behaviors,
                seed=args.seed,
                artifact_dir=_undef_artifact_dir,
            )

        if args.autodan:
            from attacks.evaluate_autodan import evaluate_autodan
            print("Stage U-2: AutoDAN on undefended model …")
            _undef_autodan = evaluate_autodan(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                tokenize_fn=model_base.tokenize_instructions_fn,
                harmful_prompts=harmful_val,
                refusal_toks=model_base.refusal_toks,
                population_size=args.autodan_population,
                num_steps=args.autodan_steps,
                n_behaviors=args.autodan_n_behaviors,
                seed=args.seed,
                artifact_dir=_undef_artifact_dir,
            )

        if args.cipherchat:
            from attacks.evaluate_cipherchat import evaluate_cipherchat
            print("Stage U-3: CipherChat on undefended model …")
            _undef_cipher = evaluate_cipherchat(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                harmful_prompts=harmful_val,
                ciphers=args.cipherchat_ciphers.split(","),
                artifact_dir=_undef_artifact_dir,
            )

        if args.pair:
            from attacks.evaluate_pair import evaluate_pair
            print("Stage U-4: PAIR on undefended model …")
            _undef_pair = evaluate_pair(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                tokenize_fn=model_base.tokenize_instructions_fn,
                harmful_prompts=harmful_val,
                refusal_toks=model_base.refusal_toks,
                n_streams=args.pair_streams,
                n_iterations=args.pair_iterations,
                n_behaviors=args.pair_n_behaviors,
                attacker_model_path=args.pair_attacker,
                seed=args.seed,
                artifact_dir=_undef_artifact_dir,
            )

        if args.renellm:
            from attacks.evaluate_renellm import evaluate_renellm
            print("Stage U-5: ReNeLLM on undefended model …")
            _undef_renellm = evaluate_renellm(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                harmful_prompts=harmful_val,
                n_rewrite_strategies=args.renellm_strategies,
                n_scenario_attempts=args.renellm_attempts,
                seed=args.seed,
                artifact_dir=_undef_artifact_dir,
            )

        if args.softopt:
            from attacks.evaluate_softopt import run_softopt_evaluation, SoftOptConfig
            print("Stage U-6: SoftOpt on undefended model …")
            device = next(model_base.model.parameters()).device
            softopt_cfg = SoftOptConfig(
                num_steps=args.softopt_steps,
                seed=args.seed,
                device=str(device),
            )
            _undef_softopt = run_softopt_evaluation(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                benchmark_path=args.softopt_benchmark,
                output_dir=_undef_artifact_dir,
                softopt_config=softopt_cfg,
                limit=args.softopt_limit,
            )

        # Write CSV row for the undefended baseline — use the same 42-column
        # schema as defended runs so all_results.csv merges cleanly.
        if args.save_csv:
            import csv as _csv
            _fieldnames = [
                "model", "defense_type", "projection_mode", "num_layers",
                "per_layer_direction", "writer_output_directions", "writer_only",
                "epsilon", "calibration_prompts", "pertinent_layers", "z_sum_norm",
                "max_cos_sim", "avg_cos_sim",
                "writer_attn_avg_cos_sim", "writer_mlp_avg_cos_sim",
                "writer_output_avg_cos_sim", "writer_attn_max_cos_sim",
                "writer_mlp_max_cos_sim", "writer_output_max_cos_sim",
                "pca8_score_undefended", "pca8_score_defended",
                "perlayer_score_undefended", "perlayer_score_defended",
                "arditi_score_undefended", "arditi_score_defended",
                "leace_score_undefended", "leace_score_defended",
                "kl_harmful", "kl_harmless",
                "harmbench_asr",
                "harmbench_asr_pre_attack",
                "harmbench_asr_post_gcg",
                "harmbench_asr_post_autodan",
                "harmbench_asr_post_cipherchat",
                "harmbench_asr_post_pair",
                "harmbench_asr_post_renellm",
                "xstest_over_refusal_rate",
                "pile_bpb",
                "alpaca_bpb",
                "alpaca_custom_bpb",
                "gsm8k_exact_match_undefended",
                "math500_exact_match_undefended",
                "mmlu_acc_undefended",
                "gsm8k_exact_match", "math500_exact_match", "mmlu_acc",
                "alpacaeval_win_rate", "alpacaeval_lc_win_rate",
                "alpacaeval_n", "alpacaeval_annotator",
                "llamaguard_asr", "softopt_asr",
                "gcg_score", "gcg_asr",
                "autodan_score", "autodan_asr",
                "cipherchat_best_asr", "cipherchat_best_cipher",
                "pair_score", "pair_asr", "renellm_asr",
            ]
            row = {k: "" for k in _fieldnames}
            row.update({
                "model":           args.model_path,
                "defense_type":    "none",
                "projection_mode": "none",
                "epsilon":         0.0,
                "calibration_prompts": 0,
                "per_layer_direction": False,
                "writer_output_directions": False,
                "writer_only": False,
                "writer_attn_avg_cos_sim": "",
                "writer_mlp_avg_cos_sim": "",
                "writer_output_avg_cos_sim": "",
                "writer_attn_max_cos_sim": "",
                "writer_mlp_max_cos_sim": "",
                "writer_output_max_cos_sim": "",
                "pca8_score_undefended":    f"{undefended_adaptive['pca']['post_attack_refusal_score']:.4f}",
                "perlayer_score_undefended": f"{undefended_adaptive['per_layer']['post_attack_refusal_score']:.4f}",
                "arditi_score_undefended":   f"{undefended_abl['arditi_refusal_score']:.4f}",
                "gcg_score":  f"{_undef_gcg['post_attack_refusal_score']:.4f}" if _undef_gcg else "",
                "gcg_asr":    f"{_undef_gcg['asr']:.4f}" if _undef_gcg else "",
                "autodan_score": f"{_undef_autodan['post_attack_refusal_score']:.4f}" if _undef_autodan else "",
                "autodan_asr":   f"{_undef_autodan['asr']:.4f}" if _undef_autodan else "",
                "cipherchat_best_asr":    f"{_undef_cipher['best_asr']:.4f}" if _undef_cipher else "",
                "cipherchat_best_cipher": _undef_cipher["best_cipher"] if _undef_cipher else "",
                "pair_score": f"{_undef_pair['post_attack_refusal_score']:.4f}" if _undef_pair else "",
                "pair_asr":   f"{_undef_pair['asr']:.4f}" if _undef_pair else "",
                "renellm_asr": f"{_undef_renellm['asr']:.4f}" if _undef_renellm else "",
                "softopt_asr": f"{_undef_softopt['softopt_asr']:.4f}" if _undef_softopt else "",
            })
            file_exists = os.path.isfile(args.save_csv)
            with open(args.save_csv, "a", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=_fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        print("\n--undefended_only: Done.")
        return

    # ==================================================================
    # Stage 2c: Pre-defense utility benchmarks
    # ==================================================================
    lm_harness_undefended = None
    if not args.skip_lm_harness:
        from benchmarks.evaluate_lm_harness import run_lm_harness
        print("=" * 60)
        print("Stage 2c: Pre-defense utility benchmarks (lm-evaluation-harness) …")
        print("=" * 60)
        try:
            _undef_lm_dir = os.path.join(obf_artifact_dir, "lm_harness_undefended")
            lm_harness_undefended = run_lm_harness(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                fwd_pre_hooks=[],
                fwd_hooks=[],
                tasks=args.lm_harness_tasks.split(","),
                n_samples=args.lm_harness_n,
                output_dir=_undef_lm_dir,
                batch_size=32,
                seed=args.seed,
            )
        except Exception as _e:
            print(f"[lm-harness undefended] WARNING: {_e}")
            lm_harness_undefended = None

    # ==================================================================
    # Stage 3: Apply defense
    # ==================================================================
    print("=" * 60)
    print(f"Stage 3: Applying defense ({args.defense_type}) …")
    print("=" * 60)

    # defense_hooks: (fwd_pre_hooks, fwd_hooks) injected into every eval call.
    # Weight-baking defenses (obfuscation) leave these empty — the model weights
    # already encode the defense.  Hook-based defenses (surgical, cast) populate
    # these and leave the weights unchanged.
    defense_fwd_pre_hooks: list = []
    defense_fwd_hooks: list = []

    if args.defense_type == "obfuscation":
        # Allow explicit layer override from CLI
        explicit_layers = (
            [int(x) for x in args.pertinent_layers.split(",")]
            if args.pertinent_layers else None
        )

        obf_result = apply_obfuscation(
            model=model_base.model,
            tokenize_fn=model_base.tokenize_instructions_fn,
            harmful_prompts=harmful_train,
            harmless_prompts=None if args.harmful_only_calibration else harmless_train,
            harmless_ratio=args.harmless_ratio,
            mean_diffs=mean_diffs,
            selected_pos=pos,
            selected_layer=layer,
            direction=direction,
            cfg=obf_cfg,
            ablation_scores=None if explicit_layers else ablation_scores,
            explicit_layers=explicit_layers,
            writer_only=args.obfuscation_writer_only,
        )
        obf_result["undefended_refusal_score"] = undefended_refusal_mean
        obf_result["undefended_arditi_score"] = undefended_abl["arditi_refusal_score"]
        obf_result["undefended_pca_attack"] = undefended_adaptive["pca"]["post_attack_refusal_score"]
        obf_result["undefended_per_layer_attack"] = undefended_adaptive["per_layer"]["post_attack_refusal_score"]
        with open(os.path.join(obf_artifact_dir, "obfuscation_result.json"), "w") as f:
            json.dump(obf_result, f, indent=4)

    elif args.defense_type == "surgical":
        surgical_result = apply_surgical(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            ablation_coeff=args.surgical_ablation_coeff,
            actadd_coeff=args.surgical_actadd_coeff,
            apply_all_layers=True,
            artifact_dir=obf_artifact_dir,
        )
        defense_fwd_pre_hooks = surgical_result["fwd_pre_hooks"]
        defense_fwd_hooks     = surgical_result["fwd_hooks"]
        # Build a minimal obf_result stub for downstream compatibility
        obf_result = {
            "pertinent_layers": list(range(model_base.model.config.num_hidden_layers)),
            "z_sum_norm": 0.0,
            "num_writers_patched": 0,
            "num_readers_patched": 0,
            "undefended_refusal_score": undefended_refusal_mean,
        }

    elif args.defense_type == "cast":
        cast_result = apply_cast(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            behavior_strength=args.cast_strength,
            condition_threshold=args.cast_threshold,
            preserve_norm=True,
            artifact_dir=obf_artifact_dir,
        )
        defense_fwd_pre_hooks = cast_result["fwd_pre_hooks"]
        defense_fwd_hooks     = cast_result["fwd_hooks"]
        obf_result = {
            "pertinent_layers": cast_result["condition_layers"],
            "z_sum_norm": 0.0,
            "num_writers_patched": 0,
            "num_readers_patched": 0,
            "undefended_refusal_score": undefended_refusal_mean,
        }

    elif args.defense_type == "alphasteer":
        from defenses.apply_alphasteer import apply_alphasteer
        as_result = apply_alphasteer(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            refusal_direction=direction,
            mean_diffs=mean_diffs,
            strength=args.alphasteer_strength,
            null_ratio=args.alphasteer_null_ratio,
            lambda_reg=args.alphasteer_lambda,
            batch_size=16,
            artifact_dir=obf_artifact_dir,
        )
        defense_fwd_pre_hooks = as_result["fwd_pre_hooks"]
        defense_fwd_hooks     = as_result["fwd_hooks"]
        obf_result = {
            "pertinent_layers": as_result["target_layers"],
            "z_sum_norm": 0.0,
            "num_writers_patched": 0,
            "num_readers_patched": 0,
            "undefended_refusal_score": undefended_refusal_mean,
        }

    elif args.defense_type == "circuit_breakers":
        cb_result = apply_circuit_breakers(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            lora_rank=args.cb_lora_rank,
            max_steps=args.cb_steps,
            lr=args.cb_lr,
            batch_size=args.cb_batch_size,
            cb_coeff_max=args.cb_coeff_max,
            retain_coeff_max=args.cb_retain_coeff_max,
            merge_weights=True,
            seed=args.seed,
            artifact_dir=obf_artifact_dir,
        )
        # After merge_and_unload the returned model replaces model_base.model
        model_base.model = cb_result["model"]
        obf_result = {
            "pertinent_layers": list(range(model_base.model.config.num_hidden_layers)),
            "z_sum_norm": 0.0,
            "num_writers_patched": 0,
            "num_readers_patched": 0,
            "undefended_refusal_score": undefended_refusal_mean,
        }

    writer_output_cos_result = None
    if writer_output_dirs_pre is not None:
        print("=" * 60)
        print("Stage 3a: Writer-output refusal-vector cosine diagnostic …")
        print("=" * 60)
        writer_output_dirs_post = collect_writer_output_refusal_directions(
            model=model_base.model,
            components=components,
            harmful_prompts=harmful_val,
            harmless_prompts=harmless_val,
            tokenize_fn=model_base.tokenize_instructions_fn,
            num_prompts=min(obf_cfg.num_calibration_prompts,
                            len(harmful_val), len(harmless_val)),
        )
        writer_output_cos_result = writer_output_direction_cosine_summary(
            reference_dirs=writer_output_dirs_pre,
            measured_dirs=writer_output_dirs_post,
            pertinent_layers=obf_result["pertinent_layers"],
        )
        print("  Writer-output avg |cos| "
              f"(pertinent): {writer_output_cos_result['writer_output_avg_cos_sim']:.4f}")
        print("  Attention avg |cos| "
              f"(pertinent): {writer_output_cos_result['writer_attn_avg_cos_sim']:.4f}")
        print("  MLP avg |cos| "
              f"(pertinent): {writer_output_cos_result['writer_mlp_avg_cos_sim']:.4f}")
        writer_cos_serialisable = {
            k: v.tolist() if isinstance(v, torch.Tensor) else v
            for k, v in writer_output_cos_result.items()
        }
        with open(os.path.join(obf_artifact_dir, "writer_output_cosine.json"), "w") as f:
            json.dump(writer_cos_serialisable, f, indent=4)

    # ==================================================================
    # Stage 3b: Post-defense integrity evaluation  (NEW)
    # ==================================================================
    print("=" * 60)
    print("Stage 3b: Evaluating defense integrity (variance + KL) …")
    print("=" * 60)

    integrity_result = evaluate_defense_integrity(
        model=model_base.model,
        components=components,
        harmful_prompts=harmful_val,
        harmless_prompts=harmless_val,
        tokenize_fn=model_base.tokenize_instructions_fn,
        pre_measurements=pre_measurements,
        num_prompts=min(obf_cfg.num_calibration_prompts, len(harmful_val)),
        pertinent_layers=obf_result["pertinent_layers"],
        artifact_dir=obf_artifact_dir,
        fwd_pre_hooks=defense_fwd_pre_hooks,
        fwd_hooks=defense_fwd_hooks,
    )

    # Serialise (filter out non-JSON-safe types)
    integrity_serialisable = {
        "kl_harmful": integrity_result["kl_harmful"],
        "kl_harmless": integrity_result["kl_harmless"],
        "summary": {
            k: v if not isinstance(v, list) else v
            for k, v in integrity_result["summary"].items()
        },
        "residual_harmful": integrity_result["residual_harmful"],
        "residual_harmless": integrity_result["residual_harmless"],
    }
    with open(os.path.join(obf_artifact_dir, "integrity_eval.json"), "w") as f:
        json.dump(integrity_serialisable, f, indent=4)

    # ==================================================================
    # Stage 4–5: Completions and loss evaluation (existing)
    # ==================================================================
    loss_evals = None
    if not args.skip_evaluations:
        try:
            from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
        except (ImportError, ModuleNotFoundError) as _vllm_err:
            print(f"[Stage 4] WARNING: could not import evaluate_jailbreak ({_vllm_err}). "
                  f"Skipping jailbreak scoring (vllm unavailable). "
                  f"HarmBench and XSTest evaluations are unaffected.")
            evaluate_jailbreak = None

        print("=" * 60)
        print("Stage 4: Evaluating completions (defended model) …")
        print("=" * 60)

        completions_dir = os.path.join(obf_artifact_dir, "completions")
        os.makedirs(completions_dir, exist_ok=True)

        for dataset_name in cfg.evaluation_datasets:
            dataset = load_dataset(dataset_name)
            completions = model_base.generate_completions(
                dataset,
                fwd_pre_hooks=defense_fwd_pre_hooks,
                fwd_hooks=defense_fwd_hooks,
                max_new_tokens=cfg.max_new_tokens,
            )
            out_path = os.path.join(completions_dir, f"{dataset_name}_defended_completions.json")
            with open(out_path, "w") as f:
                json.dump(completions, f, indent=4)

            if evaluate_jailbreak is not None:
                try:
                    evaluation = evaluate_jailbreak(
                        completions=completions,
                        methodologies=cfg.jailbreak_eval_methodologies,
                        evaluation_path=os.path.join(completions_dir, f"{dataset_name}_defended_evaluations.json"),
                    )
                    with open(os.path.join(completions_dir, f"{dataset_name}_defended_evaluations.json"), "w") as f:
                        json.dump(evaluation, f, indent=4)
                except Exception as _eval_err:
                    print(f"[Stage 4] WARNING: evaluate_jailbreak failed ({_eval_err}). Skipping.")

        print("=" * 60)
        print("Stage 5: Evaluating loss (defended model) …")
        print("=" * 60)

        loss_dir = os.path.join(obf_artifact_dir, "loss_evals")
        os.makedirs(loss_dir, exist_ok=True)

        # Generate harmless completions for on-distribution loss
        # Use harmless_val as test prompts (already loaded from mlabonne)
        harmless_test = [{"instruction": p, "category": None} for p in harmless_val[:cfg.n_test]]
        harmless_completions = model_base.generate_completions(
            harmless_test,
            fwd_pre_hooks=defense_fwd_pre_hooks,
            fwd_hooks=defense_fwd_hooks,
            max_new_tokens=cfg.max_new_tokens,
        )
        harmless_comp_path = os.path.join(completions_dir, "harmless_defended_completions.json")
        with open(harmless_comp_path, "w") as f:
            json.dump(harmless_completions, f, indent=4)

        loss_evals = evaluate_loss(
            model_base,
            fwd_pre_hooks=defense_fwd_pre_hooks,
            fwd_hooks=defense_fwd_hooks,
            batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches,
            completions_file_path=harmless_comp_path,
        )
        with open(os.path.join(loss_dir, "defended_loss_eval.json"), "w") as f:
            json.dump(loss_evals, f, indent=4)

    # ==================================================================
    # Stage 4b: HarmBench ASR evaluation (Mazeika et al. 2024)
    # ==================================================================
    harmbench_result = None
    if not args.skip_harmbench:
        from evaluations.evaluate_harmbench import evaluate_harmbench_asr

        print("=" * 60)
        print("Stage 4b: HarmBench ASR evaluation (Mazeika et al. 2024) …")
        print("=" * 60)

        try:
            harmbench_result = evaluate_harmbench_asr(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                tokenize_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=defense_fwd_pre_hooks,
                fwd_hooks=defense_fwd_hooks,
                behaviors_csv=args.harmbench_csv,
                n_behaviors=args.harmbench_n,
                seed=args.seed,
                artifact_dir=obf_artifact_dir,
            )
        except Exception as _e:
            print(f"[WARN] HarmBench evaluation failed: {_e}")
            harmbench_result = None

    # ==================================================================
    # Stage 4c: XSTest over-refusal evaluation (Röttger et al. 2023)
    # ==================================================================
    xstest_result = None
    if not args.skip_xstest:
        from benchmarks.evaluate_xstest import evaluate_xstest

        print("=" * 60)
        print("Stage 4c: XSTest over-refusal evaluation (Röttger et al. 2023) …")
        print("=" * 60)

        try:
            xstest_result = evaluate_xstest(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                tokenize_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=defense_fwd_pre_hooks,
                fwd_hooks=defense_fwd_hooks,
                artifact_dir=obf_artifact_dir,
            )
        except Exception as _e:
            print(f"[WARN] XSTest evaluation failed: {_e}")
            xstest_result = None

    # ==================================================================
    # Stage 5b: Llama Guard evaluation  (NEW)
    # ==================================================================
    llamaguard_result = None
    if args.llamaguard and not args.skip_evaluations:
        print("=" * 60)
        print("Stage 5b: Running Llama Guard evaluation …")
        print("=" * 60)

        from evaluations.evaluate_llamaguard import run_llamaguard_evaluation

        completions_path = os.path.join(
            obf_artifact_dir, "completions",
            f"{cfg.evaluation_datasets[0]}_defended_completions.json",
        )
        llamaguard_output = os.path.join(
            obf_artifact_dir, "completions", "llamaguard_evaluation.json",
        )
        llamaguard_result = run_llamaguard_evaluation(
            completions_path=completions_path,
            model_id=args.llamaguard_model,
            output_path=llamaguard_output,
        )

    # ==================================================================
    # Stage 6: Evaluate abliteration resistance  (NEW)
    # ==================================================================
    print("=" * 60)
    print("Stage 6: Evaluating abliteration resistance …")
    print("=" * 60)

    abl_result = evaluate_abliteration_resistance(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        tokenize_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        attn_modules=model_base.model_attn_modules,
        mlp_modules=model_base.model_mlp_modules,
        harmful_prompts=harmful_val,
        benign_prompts=harmless_val,
        original_direction=original_direction,
        refusal_toks=model_base.refusal_toks,
        pertinent_layers=obf_result["pertinent_layers"],
        base_fwd_pre_hooks=defense_fwd_pre_hooks,
        base_fwd_hooks=defense_fwd_hooks,
    )

    # Serialise (drop tensors)
    abl_serialisable = {
        k: v.tolist() if isinstance(v, torch.Tensor) else v
        for k, v in abl_result.items()
    }
    with open(os.path.join(obf_artifact_dir, "abliteration_eval.json"), "w") as f:
        json.dump(abl_serialisable, f, indent=4)

    # ==================================================================
    # Stage 7: Adaptive attacks  (NEW)
    # ==================================================================
    print("=" * 60)
    print("Stage 7: Running adaptive attacks …")
    print("=" * 60)

    adaptive_result = run_all_adaptive_attacks(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        tokenize_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        attn_modules=model_base.model_attn_modules,
        mlp_modules=model_base.model_mlp_modules,
        harmful_prompts=harmful_val,
        benign_prompts=harmless_val,
        original_direction=original_direction,
        refusal_toks=model_base.refusal_toks,
        pca_top_k=args.pca_top_k,
        base_fwd_pre_hooks=defense_fwd_pre_hooks,
        base_fwd_hooks=defense_fwd_hooks,
    )

    adaptive_serialisable = {
        k: {kk: vv.tolist() if isinstance(vv, torch.Tensor) else vv
            for kk, vv in v.items()}
        for k, v in adaptive_result.items()
    }
    with open(os.path.join(obf_artifact_dir, "adaptive_attacks.json"), "w") as f:
        json.dump(adaptive_serialisable, f, indent=4)

    # ==================================================================
    # Stage 7b: LEACE concept-erasure attack (Marks et al. 2023)
    # ==================================================================
    leace_result = None
    if not args.skip_leace:
        print("=" * 60)
        print("Stage 7b: Running LEACE attack (Marks et al. 2023) …")
        print("=" * 60)

        leace_result = leace_attack(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            attn_modules=model_base.model_attn_modules,
            mlp_modules=model_base.model_mlp_modules,
            harmful_prompts=harmful_val,
            benign_prompts=harmless_val,
            original_direction=original_direction,
            refusal_toks=model_base.refusal_toks,
            base_fwd_pre_hooks=defense_fwd_pre_hooks,
            base_fwd_hooks=defense_fwd_hooks,
        )

        with open(os.path.join(obf_artifact_dir, "leace_attack.json"), "w") as f:
            json.dump(leace_result, f, indent=4)

    # ==================================================================
    # Stage 7c: GCG white-box attack (Zou et al. 2023)
    # ==================================================================
    gcg_result = None
    if args.gcg:
        from attacks.evaluate_gcg import evaluate_gcg

        print("=" * 60)
        print("Stage 7c: Running GCG attack (Zou et al. 2023) …")
        print("=" * 60)

        gcg_result = evaluate_gcg(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            harmful_prompts=harmful_val,
            refusal_toks=model_base.refusal_toks,
            suffix_len=args.gcg_suffix_len,
            num_steps=args.gcg_steps,
            topk=args.gcg_topk,
            batch_size=args.gcg_batch_size,
            n_behaviors=args.gcg_n_behaviors,
            seed=args.seed,
            artifact_dir=obf_artifact_dir,
            fwd_pre_hooks=defense_fwd_pre_hooks,
            fwd_hooks=defense_fwd_hooks,
        )

    # ==================================================================
    # Stage 7d: AutoDAN-GA attack (Liu et al. 2023)
    # ==================================================================
    autodan_result = None
    if args.autodan:
        from attacks.evaluate_autodan import evaluate_autodan

        print("=" * 60)
        print("Stage 7d: Running AutoDAN-GA attack (Liu et al. 2023) …")
        print("=" * 60)

        autodan_result = evaluate_autodan(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            harmful_prompts=harmful_val,
            refusal_toks=model_base.refusal_toks,
            population_size=args.autodan_population,
            num_steps=args.autodan_steps,
            n_behaviors=args.autodan_n_behaviors,
            seed=args.seed,
            artifact_dir=obf_artifact_dir,
            fwd_pre_hooks=defense_fwd_pre_hooks,
            fwd_hooks=defense_fwd_hooks,
        )

    # ==================================================================
    # Stage 7e: CipherChat black-box attack (Yuan et al. 2023)
    # ==================================================================
    cipherchat_result = None
    if args.cipherchat:
        from attacks.evaluate_cipherchat import evaluate_cipherchat

        print("=" * 60)
        print("Stage 7e: Running CipherChat attack (Yuan et al. 2023) …")
        print("=" * 60)

        cipherchat_result = evaluate_cipherchat(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            harmful_prompts=harmful_val,
            fwd_pre_hooks=defense_fwd_pre_hooks,
            fwd_hooks=defense_fwd_hooks,
            ciphers=args.cipherchat_ciphers.split(","),
            artifact_dir=obf_artifact_dir,
        )

    # ==================================================================
    # Stage 7f: PAIR black-box attack (Chao et al. 2023)
    # ==================================================================
    pair_result = None
    if args.pair:
        from attacks.evaluate_pair import evaluate_pair

        print("=" * 60)
        print("Stage 7f: Running PAIR attack (Chao et al. 2023) …")
        print("=" * 60)

        pair_result = evaluate_pair(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            harmful_prompts=harmful_val,
            refusal_toks=model_base.refusal_toks,
            fwd_pre_hooks=defense_fwd_pre_hooks,
            fwd_hooks=defense_fwd_hooks,
            n_streams=args.pair_streams,
            n_iterations=args.pair_iterations,
            n_behaviors=args.pair_n_behaviors,
            attacker_model_path=args.pair_attacker,
            seed=args.seed,
            artifact_dir=obf_artifact_dir,
        )

    # ==================================================================
    # Stage 7g: ReNeLLM nested jailbreak attack (Ding et al. 2023)
    # ==================================================================
    renellm_result = None
    if args.renellm:
        from attacks.evaluate_renellm import evaluate_renellm

        print("=" * 60)
        print("Stage 7g: Running ReNeLLM attack (Ding et al. 2023) …")
        print("=" * 60)

        renellm_result = evaluate_renellm(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            harmful_prompts=harmful_val,
            fwd_pre_hooks=defense_fwd_pre_hooks,
            fwd_hooks=defense_fwd_hooks,
            n_rewrite_strategies=args.renellm_strategies,
            n_scenario_attempts=args.renellm_attempts,
            seed=args.seed,
            artifact_dir=obf_artifact_dir,
        )

    # ==================================================================
    # Stage 7h: SoftOpt white-box attack
    # ==================================================================
    softopt_result = None
    if args.softopt:
        print("=" * 60)
        print("Stage 7c: Running SoftOpt attack …")
        print("=" * 60)

        from attacks.evaluate_softopt import run_softopt_evaluation, SoftOptConfig

        device = next(model_base.model.parameters()).device
        softopt_cfg = SoftOptConfig(
            num_steps=args.softopt_steps,
            seed=args.seed,
            device=str(device),
        )
        softopt_result = run_softopt_evaluation(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            benchmark_path=args.softopt_benchmark,
            output_dir=obf_artifact_dir,
            softopt_config=softopt_cfg,
            limit=args.softopt_limit,
            fwd_pre_hooks=defense_fwd_pre_hooks,
            fwd_hooks=defense_fwd_hooks,
        )

    harmbench_post_attack_results = {}
    if not args.skip_harmbench:
        print("=" * 60)
        print("Stage 7i: HarmBench ASR on post-attack outputs …")
        print("=" * 60)

        try:
            if gcg_result:
                gcg_behaviors = [r["behavior"] for r in gcg_result["per_behavior"]]
                gcg_prompts = [f"{r['behavior']} {r['suffix_str']}".strip() for r in gcg_result["per_behavior"]]
                harmbench_post_attack_results["gcg"] = _generate_harmbench_for_attack(
                    label="gcg",
                    model=model_base.model,
                    tokenizer=model_base.tokenizer,
                    tokenize_fn=model_base.tokenize_instructions_fn,
                    behaviors=gcg_behaviors,
                    attacked_prompts=gcg_prompts,
                    fwd_pre_hooks=defense_fwd_pre_hooks,
                    fwd_hooks=defense_fwd_hooks,
                    max_new_tokens=cfg.max_new_tokens,
                    artifact_dir=obf_artifact_dir,
                )
            if autodan_result:
                autodan_behaviors = [r["behavior"] for r in autodan_result["per_behavior"]]
                autodan_prompts = [r["best_prompt"] for r in autodan_result["per_behavior"]]
                harmbench_post_attack_results["autodan"] = _generate_harmbench_for_attack(
                    label="autodan",
                    model=model_base.model,
                    tokenizer=model_base.tokenizer,
                    tokenize_fn=model_base.tokenize_instructions_fn,
                    behaviors=autodan_behaviors,
                    attacked_prompts=autodan_prompts,
                    fwd_pre_hooks=defense_fwd_pre_hooks,
                    fwd_hooks=defense_fwd_hooks,
                    max_new_tokens=cfg.max_new_tokens,
                    artifact_dir=obf_artifact_dir,
                )
            if cipherchat_result and cipherchat_result.get("best_cipher"):
                best_cipher = cipherchat_result["best_cipher"]
                cipher_entries = cipherchat_result["per_cipher"][best_cipher]["completions"]
                # Decode responses with the matching cipher so HarmBench (trained on
                # natural-language prompt/response pairs) sees plain text instead of
                # cipher glyphs it cannot classify. Falls back to the raw response
                # when decoding fails.
                from attacks.evaluate_cipherchat import CIPHER_SYSTEMS as _CIPHERS
                decode_fn = _CIPHERS.get(best_cipher, {}).get("decode")
                def _decode_safe(txt):
                    if decode_fn is None:
                        return txt
                    try:
                        out = decode_fn(txt)
                        return out if out and out.strip() else txt
                    except Exception:
                        return txt
                decoded_responses = [_decode_safe(c["response"]) for c in cipher_entries]
                harmbench_post_attack_results["cipherchat"] = _generate_harmbench_for_attack(
                    label="cipherchat",
                    model=model_base.model,
                    tokenizer=model_base.tokenizer,
                    tokenize_fn=model_base.tokenize_instructions_fn,
                    behaviors=[c["original_prompt"] for c in cipher_entries],
                    attacked_prompts=[c["encoded_prompt"] for c in cipher_entries],
                    responses=decoded_responses,
                    artifact_dir=obf_artifact_dir,
                )
            if pair_result:
                pair_behaviors = [r["behavior"] for r in pair_result["per_behavior"]]
                pair_prompts = [r["best_prompt"] for r in pair_result["per_behavior"]]
                harmbench_post_attack_results["pair"] = _generate_harmbench_for_attack(
                    label="pair",
                    model=model_base.model,
                    tokenizer=model_base.tokenizer,
                    tokenize_fn=model_base.tokenize_instructions_fn,
                    behaviors=pair_behaviors,
                    attacked_prompts=pair_prompts,
                    fwd_pre_hooks=defense_fwd_pre_hooks,
                    fwd_hooks=defense_fwd_hooks,
                    max_new_tokens=cfg.max_new_tokens,
                    artifact_dir=obf_artifact_dir,
                )
            if renellm_result:
                renellm_behaviors = []
                renellm_prompts = []
                renellm_responses = []
                for result in renellm_result["per_prompt"]:
                    attempts = result.get("attempts", [])
                    if not attempts:
                        continue
                    chosen = next((a for a in attempts if not a["is_refusal"]), attempts[-1])
                    renellm_behaviors.append(result["behavior"])
                    renellm_prompts.append(chosen["nested_prompt"])
                    renellm_responses.append(chosen["response"])
                harmbench_post_attack_results["renellm"] = _generate_harmbench_for_attack(
                    label="renellm",
                    model=model_base.model,
                    tokenizer=model_base.tokenizer,
                    tokenize_fn=model_base.tokenize_instructions_fn,
                    behaviors=renellm_behaviors,
                    attacked_prompts=renellm_prompts,
                    responses=renellm_responses,
                    artifact_dir=obf_artifact_dir,
                )
        except Exception as _e:
            print(f"[WARN] Post-attack HarmBench evaluation failed: {_e}")
            harmbench_post_attack_results = {}

    # ==================================================================
    # Stage 8: Heretic attack  (NEW)
    # ==================================================================
    heretic_result = None
    defended_model_path = os.path.join(obf_artifact_dir, "defended_model")
    has_inference_hooks = bool(defense_fwd_pre_hooks or defense_fwd_hooks)

    if not args.skip_heretic:
        from attacks.evaluate_heretic_attack import run_heretic_attack

        print("=" * 60)
        print("Stage 8: Saving defended model for Heretic attack …")
        print("=" * 60)

        if has_inference_hooks:
            print("[WARN] Skipping Heretic for hook-based defenses; saved weights do not encode runtime hooks.")
        else:
            model_base.model.save_pretrained(defended_model_path)
            model_base.tokenizer.save_pretrained(defended_model_path)
            print(f"  Saved to: {defended_model_path}")

            print("=" * 60)
            print(f"Stage 8: Running Heretic attack ({args.heretic_trials} trials) …")
            print("=" * 60)

            heretic_result = run_heretic_attack(
                defended_model_path=defended_model_path,
                artifact_dir=obf_artifact_dir,
                n_trials=args.heretic_trials,
                system_prompt=SYSTEM_PROMPT,
            )

            with open(os.path.join(obf_artifact_dir, "heretic_attack.json"), "w") as f:
                json.dump(heretic_result, f, indent=4)

    # ==================================================================
    # Stage 9: Utility benchmarks — GSM8k, MATH500, MMLU (lm-eval-harness)
    # ==================================================================
    lm_harness_result = None
    if not args.skip_lm_harness:
        from benchmarks.evaluate_lm_harness import run_lm_harness

        print("=" * 60)
        print("Stage 9: Utility benchmarks (lm-evaluation-harness) …")
        print("=" * 60)

        if not has_inference_hooks:
            # Keep a saved copy for external tools and debugging.
            import shutil as _shutil
            if os.path.isdir(defended_model_path):
                _shutil.rmtree(defended_model_path)
            print(f"[lm-harness] Saving defended model to {defended_model_path} …")
            model_base.model.save_pretrained(defended_model_path)
            model_base.tokenizer.save_pretrained(defended_model_path)

        try:
            lm_harness_result = run_lm_harness(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                fwd_pre_hooks=defense_fwd_pre_hooks,
                fwd_hooks=defense_fwd_hooks,
                tasks=args.lm_harness_tasks.split(","),
                n_samples=args.lm_harness_n,
                output_dir=os.path.join(obf_artifact_dir, "lm_harness"),
                batch_size=32,
                seed=args.seed,
            )
        except Exception as _e:
            print(f"[WARN] lm-harness evaluation failed: {_e}")
            lm_harness_result = None

    # ==================================================================
    # Stage 9b: AlpacaEval — generation-quality / instruction-following
    # ==================================================================
    alpacaeval_result = None
    if not args.skip_alpacaeval:
        print("=" * 60)
        print("Stage 9b: AlpacaEval generation-quality evaluation …")
        print("=" * 60)

        try:
            from benchmarks.evaluate_alpacaeval import evaluate_alpacaeval

            alpacaeval_result = evaluate_alpacaeval(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                tokenize_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=defense_fwd_pre_hooks,
                fwd_hooks=defense_fwd_hooks,
                n_samples=args.alpacaeval_n,
                max_new_tokens=args.alpacaeval_max_new_tokens,
                batch_size=32,
                seed=args.seed,
                run_judge=not args.alpacaeval_skip_judge,
                annotators_config=args.alpacaeval_annotator,
                generator_name=f"aprs_{obf_cfg.projection_mode}_eps{obf_cfg.epsilon}",
                artifact_dir=os.path.join(obf_artifact_dir, "alpacaeval"),
            )
        except Exception as _e:
            print(f"[WARN] AlpacaEval evaluation failed: {_e}")
            alpacaeval_result = None

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — Summary")
    print("=" * 60)
    print(f"  Model              : {args.model_path}")
    print(f"  Projection mode    : {obf_cfg.projection_mode}")
    print(f"  Per-layer direction: {obf_cfg.per_layer_direction}")
    print(f"  epsilon            : {obf_cfg.epsilon}")
    print(f"  Pertinent layers   : {obf_result['pertinent_layers']}")
    print(f"  z_sum norm         : {obf_result['z_sum_norm']:.4f}")
    print(f"  Writers patched    : {obf_result['num_writers_patched']}")
    print(f"  Readers patched    : {obf_result['num_readers_patched']}")

    s = integrity_result["summary"]
    print(f"  --- Integrity (variance + KL) ---")
    print(f"  Harmful  mean |Δσ²/σ²| : "
          f"attn={s.get('harmful_attn_mean_abs_var_shift', float('nan')):.6f}, "
          f"mlp={s.get('harmful_mlp_mean_abs_var_shift', float('nan')):.6f}")
    print(f"  Harmless mean |Δσ²/σ²| : "
          f"attn={s.get('harmless_attn_mean_abs_var_shift', float('nan')):.6f}, "
          f"mlp={s.get('harmless_mlp_mean_abs_var_shift', float('nan')):.6f}")
    print(f"  Harmful  mean cos_sim  : "
          f"attn={s.get('harmful_attn_mean_cos_sim', float('nan')):.6f}, "
          f"mlp={s.get('harmful_mlp_mean_cos_sim', float('nan')):.6f}")
    print(f"  Harmless mean cos_sim  : "
          f"attn={s.get('harmless_attn_mean_cos_sim', float('nan')):.6f}, "
          f"mlp={s.get('harmless_mlp_mean_cos_sim', float('nan')):.6f}")
    print(f"  KL(orig||def) harmful  : {integrity_result['kl_harmful']['kl_forward_mean']:.6f}")
    print(f"  KL(orig||def) harmless : {integrity_result['kl_harmless']['kl_forward_mean']:.6f}")
    print(f"  JSD harmful            : {integrity_result['kl_harmful']['jsd_mean']:.6f}")
    print(f"  JSD harmless           : {integrity_result['kl_harmless']['jsd_mean']:.6f}")

    print(f"  --- Abliteration resistance ---")
    print(f"                        Undefended → Defended")
    print(f"  Refusal score      : {undefended_refusal_mean:.4f} → {abl_result['baseline_refusal_score']:.4f}")
    print(f"  Arditi attack  : {undefended_abl['arditi_refusal_score']:.4f} → {abl_result['arditi_refusal_score']:.4f}")
    print(f"  PCA-{args.pca_top_k} attack      : {undefended_adaptive['pca']['post_attack_refusal_score']:.4f} → {adaptive_result['pca']['post_attack_refusal_score']:.4f}")
    print(f"  Per-layer attack   : {undefended_adaptive['per_layer']['post_attack_refusal_score']:.4f} → {adaptive_result['per_layer']['post_attack_refusal_score']:.4f}")
    print(f"  Max |cos_sim|      : {abl_result['max_cos_sim']:.4f}")
    print(f"  Avg |cos_sim| (pertinent): {abl_result['mean_cos_sim']:.4f}")

    if leace_result:
        undef_leace = undefended_leace['post_attack_refusal_score'] if undefended_leace else float('nan')
        print(f"  --- LEACE attack (Marks et al.) ---")
        print(f"  LEACE score        : {undef_leace:.4f} → {leace_result['post_attack_refusal_score']:.4f}")
        print(f"  LEACE max |cos_sim|: {leace_result['max_cos_sim']:.4f}")

    if gcg_result:
        print(f"  --- GCG attack (Zou et al.) ---")
        print(f"  Post-attack refusal: {gcg_result['post_attack_refusal_score']:.4f}")
        print(f"  ASR (target loss)  : {gcg_result['asr']:.4f}")

    if autodan_result:
        print(f"  --- AutoDAN-GA attack (Liu et al.) ---")
        print(f"  Post-attack refusal: {autodan_result['post_attack_refusal_score']:.4f}")
        print(f"  ASR (target loss)  : {autodan_result['asr']:.4f}")

    if cipherchat_result:
        print(f"  --- CipherChat attack (Yuan et al.) ---")
        print(f"  Best ASR           : {cipherchat_result['best_asr']:.4f} "
              f"({cipherchat_result['best_cipher']})")

    if pair_result:
        print(f"  --- PAIR attack (Chao et al.) ---")
        print(f"  Post-attack refusal: {pair_result['post_attack_refusal_score']:.4f}")
        print(f"  ASR (score=10)     : {pair_result['asr']:.4f}")

    if renellm_result:
        print(f"  --- ReNeLLM attack (Ding et al.) ---")
        print(f"  ASR                : {renellm_result['asr']:.4f} "
              f"({renellm_result['n_jailbroken']}/{renellm_result['n_total']})")

    if softopt_result:
        print(f"  --- SoftOpt attack ---")
        print(f"  ASR (HarmBench)    : {softopt_result['softopt_asr']:.4f} "
              f"({softopt_result['num_harmful']}/{softopt_result['total']})")
        print(f"  Avg final loss     : {softopt_result['avg_final_loss']:.4f}")

    if harmbench_result:
        print(f"  --- HarmBench ASR (Mazeika et al.) ---")
        print(f"  ASR                : {harmbench_result['asr']:.4f} "
              f"({harmbench_result['n_harmful']}/{harmbench_result['n_total']})")
    for attack_name, attack_result in harmbench_post_attack_results.items():
        if attack_result:
            print(f"  HarmBench after {attack_name:9s}: {attack_result['asr']:.4f} "
                  f"({attack_result['n_harmful']}/{attack_result['n_total']})")

    if xstest_result:
        print(f"  --- XSTest over-refusal (Röttger et al.) ---")
        print(f"  Over-refusal rate  : {xstest_result['over_refusal_rate']:.4f} "
              f"({xstest_result['n_refused']}/{xstest_result['n_total']})")

    if lm_harness_result:
        print(f"  --- Utility benchmarks (lm-evaluation-harness) ---")
        for task, metrics in lm_harness_result.items():
            print(f"  {task:12s}: {metrics}")

    if alpacaeval_result:
        print(f"  --- AlpacaEval (generation quality) ---")
        print(f"  n_samples          : {alpacaeval_result.get('n_samples')}")
        wr = alpacaeval_result.get("win_rate")
        lc = alpacaeval_result.get("length_controlled_win_rate")
        ann = alpacaeval_result.get("annotator")
        if wr is not None:
            print(f"  win_rate           : {wr:.4f}")
            print(f"  lc_win_rate        : {lc:.4f}")
            print(f"  annotator          : {ann}")
        else:
            print(f"  (judging skipped — completions saved to "
                  f"{alpacaeval_result.get('completions_path')})")

    if llamaguard_result:
        print(f"  --- Llama Guard ---")
        print(f"  ASR (unsafe rate)  : {llamaguard_result['llamaguard_asr']:.4f} "
              f"({llamaguard_result['num_unsafe']}/{llamaguard_result['total']})")

    if heretic_result:
        print(f"  --- Heretic attack ---")
        print(f"  Base refusals      : {heretic_result['base_refusals']}/{heretic_result['total_bad_prompts']} "
              f"({heretic_result['base_refusals_pct']:.1f}%)")
        print(f"  Best refusals      : {heretic_result['best_refusals']}/{heretic_result['total_bad_prompts']} "
              f"({heretic_result['best_refusals_pct']:.1f}%)")
        print(f"  Best KL            : {heretic_result['best_kl']:.4f}")

    print(f"  Artifacts          : {obf_artifact_dir}")

    # ==================================================================
    # Save results to CSV
    # ==================================================================
    if args.save_csv:
        import csv

        csv_path = args.save_csv
        fieldnames = [
            "model", "defense_type", "projection_mode", "num_layers",
            "per_layer_direction", "writer_output_directions", "writer_only",
            "epsilon", "calibration_prompts", "pertinent_layers", "z_sum_norm",
            "max_cos_sim", "avg_cos_sim",
            "writer_attn_avg_cos_sim", "writer_mlp_avg_cos_sim",
            "writer_output_avg_cos_sim", "writer_attn_max_cos_sim",
            "writer_mlp_max_cos_sim", "writer_output_max_cos_sim",
            "pca8_score_undefended", "pca8_score_defended",
            "perlayer_score_undefended", "perlayer_score_defended",
            "arditi_score_undefended", "arditi_score_defended",
            "leace_score_undefended", "leace_score_defended",
            "kl_harmful", "kl_harmless",
            "harmbench_asr",
            "harmbench_asr_pre_attack",
            "harmbench_asr_post_gcg",
            "harmbench_asr_post_autodan",
            "harmbench_asr_post_cipherchat",
                "harmbench_asr_post_pair",
                "harmbench_asr_post_renellm",
                "xstest_over_refusal_rate",
                "pile_bpb",
                "alpaca_bpb",
                "alpaca_custom_bpb",
                "gsm8k_exact_match_undefended",
                "math500_exact_match_undefended",
                "mmlu_acc_undefended",
            "gsm8k_exact_match",
            "math500_exact_match",
            "mmlu_acc",
            "alpacaeval_win_rate",
            "alpacaeval_lc_win_rate",
            "alpacaeval_n",
            "alpacaeval_annotator",
            "llamaguard_asr",
            "softopt_asr",
            "gcg_score",
            "gcg_asr",
            "autodan_score",
            "autodan_asr",
            "cipherchat_best_asr",
            "cipherchat_best_cipher",
            "pair_score",
            "pair_asr",
            "renellm_asr",
        ]

        row = {
            "model": args.model_path,
            "defense_type": args.defense_type,
            "projection_mode": obf_cfg.projection_mode if args.defense_type == "obfuscation" else "—",
            "num_layers": len(obf_result["pertinent_layers"]),
            "per_layer_direction": obf_cfg.per_layer_direction,
            "writer_output_directions": obf_cfg.writer_output_directions,
            "writer_only": bool(args.obfuscation_writer_only),
            "epsilon": obf_cfg.epsilon,
            "calibration_prompts": obf_cfg.num_calibration_prompts,
            "pertinent_layers": str(obf_result["pertinent_layers"]),
            "z_sum_norm": f"{obf_result['z_sum_norm']:.4f}",
            "max_cos_sim": f"{abl_result['max_cos_sim']:.4f}",
            "avg_cos_sim": f"{abl_result['mean_cos_sim']:.4f}",
            "writer_attn_avg_cos_sim": (
                f"{writer_output_cos_result['writer_attn_avg_cos_sim']:.4f}"
                if writer_output_cos_result else ""
            ),
            "writer_mlp_avg_cos_sim": (
                f"{writer_output_cos_result['writer_mlp_avg_cos_sim']:.4f}"
                if writer_output_cos_result else ""
            ),
            "writer_output_avg_cos_sim": (
                f"{writer_output_cos_result['writer_output_avg_cos_sim']:.4f}"
                if writer_output_cos_result else ""
            ),
            "writer_attn_max_cos_sim": (
                f"{writer_output_cos_result['writer_attn_max_cos_sim']:.4f}"
                if writer_output_cos_result else ""
            ),
            "writer_mlp_max_cos_sim": (
                f"{writer_output_cos_result['writer_mlp_max_cos_sim']:.4f}"
                if writer_output_cos_result else ""
            ),
            "writer_output_max_cos_sim": (
                f"{writer_output_cos_result['writer_output_max_cos_sim']:.4f}"
                if writer_output_cos_result else ""
            ),
            "pca8_score_undefended": f"{undefended_adaptive['pca']['post_attack_refusal_score']:.4f}",
            "pca8_score_defended": f"{adaptive_result['pca']['post_attack_refusal_score']:.4f}",
            "perlayer_score_undefended": f"{undefended_adaptive['per_layer']['post_attack_refusal_score']:.4f}",
            "perlayer_score_defended": f"{adaptive_result['per_layer']['post_attack_refusal_score']:.4f}",
            "arditi_score_undefended": f"{undefended_abl['arditi_refusal_score']:.4f}",
            "arditi_score_defended": f"{abl_result['arditi_refusal_score']:.4f}",
            "kl_harmful": f"{integrity_result['kl_harmful']['kl_forward_mean']:.6f}",
            "kl_harmless": f"{integrity_result['kl_harmless']['kl_forward_mean']:.6f}",
            "leace_score_undefended": "",
            "leace_score_defended": "",
            "harmbench_asr": "",
            "harmbench_asr_pre_attack": "",
            "harmbench_asr_post_gcg": "",
            "harmbench_asr_post_autodan": "",
            "harmbench_asr_post_cipherchat": "",
            "harmbench_asr_post_pair": "",
            "harmbench_asr_post_renellm": "",
            "xstest_over_refusal_rate": "",
            "pile_bpb": "",
            "alpaca_bpb": "",
            "alpaca_custom_bpb": "",
            "gsm8k_exact_match": "",
            "math500_exact_match": "",
            "mmlu_acc": "",
            "alpacaeval_win_rate": "",
            "alpacaeval_lc_win_rate": "",
            "alpacaeval_n": "",
            "alpacaeval_annotator": "",
            "llamaguard_asr": "",
            "softopt_asr": "",
            "gcg_score": "",
            "gcg_asr": "",
            "autodan_score": "",
            "autodan_asr": "",
            "cipherchat_best_asr": "",
            "cipherchat_best_cipher": "",
            "pair_score": "",
            "pair_asr": "",
            "renellm_asr": "",
        }

        if undefended_leace:
            row["leace_score_undefended"] = f"{undefended_leace['post_attack_refusal_score']:.4f}"
        if leace_result:
            row["leace_score_defended"] = f"{leace_result['post_attack_refusal_score']:.4f}"
        if harmbench_result:
            row["harmbench_asr"] = f"{harmbench_result['asr']:.4f}"
            row["harmbench_asr_pre_attack"] = f"{harmbench_result['asr']:.4f}"
        if harmbench_post_attack_results.get("gcg"):
            row["harmbench_asr_post_gcg"] = f"{harmbench_post_attack_results['gcg']['asr']:.4f}"
        if harmbench_post_attack_results.get("autodan"):
            row["harmbench_asr_post_autodan"] = f"{harmbench_post_attack_results['autodan']['asr']:.4f}"
        if harmbench_post_attack_results.get("cipherchat"):
            row["harmbench_asr_post_cipherchat"] = f"{harmbench_post_attack_results['cipherchat']['asr']:.4f}"
        if harmbench_post_attack_results.get("pair"):
            row["harmbench_asr_post_pair"] = f"{harmbench_post_attack_results['pair']['asr']:.4f}"
        if harmbench_post_attack_results.get("renellm"):
            row["harmbench_asr_post_renellm"] = f"{harmbench_post_attack_results['renellm']['asr']:.4f}"
        if xstest_result:
            row["xstest_over_refusal_rate"] = f"{xstest_result['over_refusal_rate']:.4f}"
        if loss_evals:
            if "pile" in loss_evals and loss_evals["pile"].get("bpb") is not None:
                row["pile_bpb"] = f"{loss_evals['pile']['bpb']:.4f}"
            if "alpaca" in loss_evals and loss_evals["alpaca"].get("bpb") is not None:
                row["alpaca_bpb"] = f"{loss_evals['alpaca']['bpb']:.4f}"
            if ("alpaca_custom_completions" in loss_evals
                    and loss_evals["alpaca_custom_completions"].get("bpb") is not None):
                row["alpaca_custom_bpb"] = (
                    f"{loss_evals['alpaca_custom_completions']['bpb']:.4f}"
                )
        if lm_harness_undefended:
            if "gsm8k" in lm_harness_undefended:
                v = lm_harness_undefended["gsm8k"].get("exact_match")
                if v is not None:
                    row["gsm8k_exact_match_undefended"] = f"{v:.4f}"
            if "math500" in lm_harness_undefended:
                v = lm_harness_undefended["math500"].get("exact_match")
                if v is not None:
                    row["math500_exact_match_undefended"] = f"{v:.4f}"
            if "mmlu" in lm_harness_undefended:
                v = lm_harness_undefended["mmlu"].get("acc")
                if v is not None:
                    row["mmlu_acc_undefended"] = f"{v:.4f}"
        if lm_harness_result:
            if "gsm8k" in lm_harness_result:
                v = lm_harness_result["gsm8k"].get("exact_match")
                if v is not None:
                    row["gsm8k_exact_match"] = f"{v:.4f}"
            if "math500" in lm_harness_result:
                v = lm_harness_result["math500"].get("exact_match")
                if v is not None:
                    row["math500_exact_match"] = f"{v:.4f}"
            if "mmlu" in lm_harness_result:
                v = lm_harness_result["mmlu"].get("acc")
                if v is not None:
                    row["mmlu_acc"] = f"{v:.4f}"

        if alpacaeval_result:
            row["alpacaeval_n"] = str(alpacaeval_result.get("n_samples", ""))
            wr = alpacaeval_result.get("win_rate")
            lc = alpacaeval_result.get("length_controlled_win_rate")
            ann = alpacaeval_result.get("annotator")
            if wr is not None:
                row["alpacaeval_win_rate"] = f"{wr:.4f}"
            if lc is not None:
                row["alpacaeval_lc_win_rate"] = f"{lc:.4f}"
            if ann:
                row["alpacaeval_annotator"] = ann

        if llamaguard_result:
            row["llamaguard_asr"] = f"{llamaguard_result['llamaguard_asr']:.4f}"
        if softopt_result:
            row["softopt_asr"] = f"{softopt_result['softopt_asr']:.4f}"
        if gcg_result:
            row["gcg_score"] = f"{gcg_result['post_attack_refusal_score']:.4f}"
            row["gcg_asr"]   = f"{gcg_result['asr']:.4f}"
        if autodan_result:
            row["autodan_score"] = f"{autodan_result['post_attack_refusal_score']:.4f}"
            row["autodan_asr"]   = f"{autodan_result['asr']:.4f}"
        if cipherchat_result:
            row["cipherchat_best_asr"]    = f"{cipherchat_result['best_asr']:.4f}"
            row["cipherchat_best_cipher"] = cipherchat_result["best_cipher"] or ""
        if pair_result:
            row["pair_score"] = f"{pair_result['post_attack_refusal_score']:.4f}"
            row["pair_asr"]   = f"{pair_result['asr']:.4f}"
        if renellm_result:
            row["renellm_asr"] = f"{renellm_result['asr']:.4f}"

        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print(f"  Results appended   : {csv_path}")


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)
