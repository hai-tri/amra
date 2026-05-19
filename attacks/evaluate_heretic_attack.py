"""
Heretic adaptive attack against the defended model.

Runs Heretic's Optuna-based abliteration optimizer against a saved copy of
the defended model.  Heretic co-minimises refusal count and KL divergence,
making it a strictly stronger attack than standard single-direction abliteration.

Usage (standalone):
    python evaluate_heretic_attack.py \\
        --model_path /path/to/saved/defended/model \\
        --n_trials 50

Or called from run_obfuscation_pipeline.py as Stage 8.
"""

import math
import os
import sys
import time
import warnings
from dataclasses import asdict
from os.path import commonprefix
from pathlib import Path
from typing import Dict, List, Optional

import optuna
import torch
import torch.nn.functional as F

# Add heretic src to path
_HERETIC_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "heretic", "src")
if _HERETIC_SRC not in sys.path:
    sys.path.insert(0, _HERETIC_SRC)

from heretic.config import DatasetSpecification, Settings
from heretic.evaluator import Evaluator
from heretic.model import AbliterationParameters, Model
from heretic.utils import load_prompts

from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import StudyDirection
from optuna import TrialPruned
from optuna.trial import Trial, TrialState


def run_heretic_attack(
    defended_model_path: str,
    artifact_dir: str,
    n_trials: int = 50,
    n_startup_trials: int = 15,
    system_prompt: str = "You are a helpful assistant.",
    train_samples: int = 400,
    eval_samples: int = 100,
    max_response_length: int = 100,
) -> Dict:
    """
    Run Heretic's Optuna-based abliteration against the saved defended model.

    Parameters
    ----------
    defended_model_path : str
        Path to the saved defended model (output of save_defended_model).
    artifact_dir : str
        Directory to save checkpoints and results.
    n_trials : int
        Number of Optuna trials.  50 is enough for a good attack estimate.
    n_startup_trials : int
        Number of random exploration trials before TPE takes over.
    system_prompt : str
        System prompt used during evaluation.

    Returns
    -------
    dict with keys:
        * base_refusals     — refusals before any abliteration
        * best_refusals     — lowest refusals achieved across all trials
        * best_kl           — KL divergence at the best trial
        * n_trials_run      — number of trials actually completed
        * refusals_per_trial — list of refusal counts per trial
    """
    # Silence noise
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    # Build Settings programmatically — bypass CLI parsing by patching sys.argv
    orig_argv = sys.argv
    sys.argv = ["heretic", "--model", defended_model_path]
    settings = Settings(
        model=defended_model_path,
        n_trials=n_trials,
        n_startup_trials=n_startup_trials,
        system_prompt=system_prompt,
        max_response_length=max_response_length,
        batch_size=1,
        print_responses=False,
        # Use same datasets as the rest of our pipeline
        good_prompts=DatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split=f"train[:{train_samples}]",
            column="text",
        ),
        bad_prompts=DatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split=f"train[:{train_samples}]",
            column="text",
        ),
        good_evaluation_prompts=DatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split=f"test[:{eval_samples}]",
            column="text",
        ),
        bad_evaluation_prompts=DatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split=f"test[:{eval_samples}]",
            column="text",
        ),
        study_checkpoint_dir=os.path.join(artifact_dir, "heretic_checkpoints"),
    )
    sys.argv = orig_argv

    # Load model via Heretic's Model class
    print("[heretic attack] Loading defended model …")
    model = Model(settings)

    # Load prompts
    print("[heretic attack] Loading prompts …")
    good_prompts = load_prompts(settings, settings.good_prompts)
    bad_prompts = load_prompts(settings, settings.bad_prompts)

    # Detect common response prefix (for CoT models)
    prefix_n = min(50, len(good_prompts), len(bad_prompts))
    prefix_check_prompts = good_prompts[:prefix_n] + bad_prompts[:prefix_n]
    responses = model.get_responses_batched(prefix_check_prompts)
    model.response_prefix = commonprefix(responses).rstrip(" ")

    # Set up evaluator — this also counts base refusals
    print("[heretic attack] Setting up evaluator …")
    evaluator = Evaluator(settings, model)
    base_refusals = evaluator.base_refusals
    total_bad = len(evaluator.bad_prompts)
    print(f"[heretic attack] Base refusals (defended model, no attack): "
          f"{base_refusals}/{total_bad}")

    # Compute refusal directions
    print("[heretic attack] Computing per-layer refusal directions …")
    good_residuals = model.get_residuals_batched(good_prompts)
    bad_residuals = model.get_residuals_batched(bad_prompts)
    good_means = good_residuals.mean(dim=0)
    bad_means = bad_residuals.mean(dim=0)
    refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)
    del good_residuals, bad_residuals

    # Optuna study
    os.makedirs(settings.study_checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(
        settings.study_checkpoint_dir, "heretic_attack.jsonl"
    )
    lock_obj = JournalFileOpenLock(checkpoint_file)
    backend = JournalFileBackend(checkpoint_file, lock_obj=lock_obj)
    storage = JournalStorage(backend)

    study = optuna.create_study(
        sampler=TPESampler(
            n_startup_trials=n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
        ),
        directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
        storage=storage,
        study_name="heretic_attack",
        load_if_exists=True,
    )

    trial_index = 0
    start_time = time.perf_counter()
    refusals_per_trial = []

    def objective(trial: Trial) -> tuple[float, float]:
        nonlocal trial_index
        trial_index += 1

        direction_scope = trial.suggest_categorical(
            "direction_scope", ["global", "per layer"]
        )
        last_layer_index = len(model.get_layers()) - 1

        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * last_layer_index,
            0.9 * last_layer_index,
        )
        if direction_scope == "per layer":
            direction_index = None

        parameters = {}
        for component in model.get_abliterable_components():
            max_weight = trial.suggest_float(f"{component}.max_weight", 0.8, 1.5)
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,
            )
            min_weight = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance", 1.0, 0.6 * last_layer_index
            )
            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=min_weight * max_weight,
                min_weight_distance=min_weight_distance,
            )

        model.reset_model()
        model.abliterate(refusal_directions, direction_index, parameters)
        score, kl_divergence, refusals = evaluator.get_score()

        refusals_per_trial.append(refusals)
        elapsed = time.perf_counter() - start_time
        print(f"[heretic attack] Trial {trial_index}/{n_trials} — "
              f"refusals={refusals}/{total_bad}, KL={kl_divergence:.4f}, "
              f"elapsed={elapsed:.0f}s")

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)
        return score

    def objective_wrapper(trial: Trial) -> tuple[float, float]:
        try:
            return objective(trial)
        except KeyboardInterrupt:
            trial.study.stop()
            raise TrialPruned()

    print(f"\n[heretic attack] Running {n_trials} trials …")
    try:
        study.optimize(objective_wrapper, n_trials=n_trials)
    except KeyboardInterrupt:
        pass

    # Find best result (fewest refusals, then lowest KL)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        print("[heretic attack] No trials completed.")
        return {"base_refusals": base_refusals, "best_refusals": None}

    best = min(completed, key=lambda t: (t.user_attrs["refusals"], t.user_attrs["kl_divergence"]))
    best_refusals = best.user_attrs["refusals"]
    best_kl = best.user_attrs["kl_divergence"]

    print(f"\n[heretic attack] === Results ===")
    print(f"  Base refusals (defended, no attack) : {base_refusals}/{total_bad} "
          f"({100*base_refusals/total_bad:.1f}%)")
    print(f"  Best refusals (after Heretic attack) : {best_refusals}/{total_bad} "
          f"({100*best_refusals/total_bad:.1f}%)")
    print(f"  Best KL divergence                   : {best_kl:.4f}")
    print(f"  Trials completed                     : {len(completed)}/{n_trials}")

    return {
        "base_refusals": base_refusals,
        "base_refusals_pct": 100 * base_refusals / total_bad,
        "best_refusals": best_refusals,
        "best_refusals_pct": 100 * best_refusals / total_bad,
        "best_kl": best_kl,
        "n_trials_run": len(completed),
        "total_bad_prompts": total_bad,
        "refusals_per_trial": refusals_per_trial,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--artifact_dir", default="heretic_output")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--train_samples", type=int, default=400)
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--max_response_length", type=int, default=100)
    args = parser.parse_args()

    run_heretic_attack(
        defended_model_path=args.model_path,
        artifact_dir=args.artifact_dir,
        n_trials=args.n_trials,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        max_response_length=args.max_response_length,
    )
