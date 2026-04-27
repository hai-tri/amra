from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ObfuscationConfig:
    # Random alias vector standard deviation. Controls pollution magnitude.
    # Smaller = less LayerNorm distortion, larger = harder for attacker to filter out.
    epsilon: float = 0.1

    # Number of pertinent layers to patch.
    # None = auto-detect from data (layers whose refusal magnitude > 20% of peak).
    # Set to an int to override with a fixed top-k (useful for ablation sweeps).
    num_pertinent_layers: Optional[int] = None

    # Number of harmful prompts used for calibration forward passes.
    # More = better generalization of rank-one patches.
    num_calibration_prompts: int = 32

    # Number of prompts used for each empirical probe of the residual stream
    # during iterative patching.  Each layer triggers up to 2 probes (one before
    # attention sublayer patches, one between attn and MLP sublayers), so cost
    # scales as ~2 * num_layers * num_probe_prompts forward passes.  Subset of
    # the calibration prompts — smaller than calibration to keep cost bounded.
    num_probe_prompts: int = 8

    # Whether to use separate random alias vectors for W_O and W_down at each layer.
    # True = more obfuscation (2 random vectors per layer), False = shared alias.
    separate_attn_mlp_aliases: bool = True

    # Random seed for reproducible alias generation.
    seed: int = 42

    # Which writer matrices to patch at pertinent layers.
    # Options: "both", "attn_only", "mlp_only"
    patch_writers: str = "both"

    # Projection mode for writer patches.
    # "hadamard"  = replace r̂ component with r̂ ⊙ ξ, ξ ~ N(0, ε²I).
    #               Element-wise Gaussian noise weighted by r̂.
    # "binary"    = replace r̂ component with r̂ ⊙ s, s_i ∈ {-1, +1}.
    #               Rademacher sign flips. Magnitude = ||r̂|| = 1. ε not used.
    # "mask"      = replace r̂ component with r̂ ⊙ m, m_i ∈ {0, 1}.
    #               Random dropout mask. Magnitude ≈ ||r̂||/√2. ε not used.
    # "scalar_projection"  = replace r̂ component with η · r̂ (single random scalar).
    #               Pollution purely along r̂.
    # "full"      = replace the refusal component with a full random alias vector
    #               (original behaviour).  Highest pollution, worst utility.
    projection_mode: str = "hadamard"

    # Use per-layer refusal directions instead of the global r̂.
    # When True, each writer patch uses mean_diffs[pos, layer] as r̂ for that
    # layer, so a different direction is obfuscated at each layer.  The attacker
    # must then recover a different direction per layer rather than one global one.
    per_layer_direction: bool = False

    # Use writer-output refusal directions for writer patches.
    # When True, APRS estimates a separate harmful-minus-harmless direction at
    # each attention output projection (W_O) and each MLP output projection
    # (W_down), then patches each writer using its own local output-space
    # direction.  If a local direction is degenerate, the code falls back to
    # the residual-stream direction selected by ``per_layer_direction`` /
    # global r̂.
    writer_output_directions: bool = False

    # Number of writer-output refusal directions to patch per writer.  The
    # default preserves the existing rank-1 behavior.  Values >1 require
    # ``writer_output_directions=True`` and extract a PCA subspace from
    # harmful-vs-harmless writer outputs at each pertinent layer.
    num_writer_directions: int = 1
