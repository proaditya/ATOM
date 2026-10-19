# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The deployments the Qwen3.8-27B megakernel serves, and why any other is
refused."""

import os

from atom.models.qwen38_27b.mono import dims as d
from atom.mono.runtime.deployment import first_refusal
from atom.utils import envs

ARCHITECTURES = ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
# the kernel builds a deployment loads (``kernels.build.VARIANTS``):
# ``ATOM_QWEN38_MK_KERNELS``, default the 8-row ``m8``; ``m16`` adds 16-row launches
DEFAULT_KERNELS = ("m8",)
# the 8-row build's split layout caps one attention split at 128 pages (4096
# positions), which a position past 131064 needs more than
MAX_MODEL_LEN = 131064


def kernel_names() -> tuple[str, ...]:
    from atom.models.qwen38_27b.mono.kernels.build import VARIANTS

    names = tuple(
        x
        for x in os.environ.get(
            "ATOM_QWEN38_MK_KERNELS", ",".join(DEFAULT_KERNELS)
        ).split(",")
        if x
    )
    unknown = [n for n in names if n not in VARIANTS]
    if unknown or len({VARIANTS[n].rows for n in names}) != len(names):
        raise ValueError(
            f"ATOM_QWEN38_MK_KERNELS={names}: one build a width, of {sorted(VARIANTS)}"
        )
    return names


def max_spec_tokens() -> int:
    """A launch's rows are consecutive positions of one sequence: the anchor
    token and the drafted ones, at most the widest build's rows."""
    from atom.models.qwen38_27b.mono.kernels.build import VARIANTS

    return max(VARIANTS[n].rows for n in kernel_names()) - 1


def _text(hf):
    return getattr(hf, "text_config", hf)


def model_matches(hf) -> bool:
    """Is this checkpoint the dense 27B the kernel is built for?"""
    t = _text(hf)
    rope = getattr(t, "rope_parameters", None) or {}
    pattern = [
        "full_attention" if d.is_attention(i) else "linear_attention"
        for i in range(d.LAYERS)
    ]
    return (
        (getattr(hf, "architectures", None) or [""])[0] in ARCHITECTURES
        and getattr(t, "model_type", None) == "qwen3_5_text"
        and t.hidden_size == d.HIDDEN
        and t.num_hidden_layers == d.LAYERS
        and list(t.layer_types) == pattern
        and t.vocab_size == d.VOCAB
        and t.intermediate_size == d.INTERMEDIATE
        and (t.num_attention_heads, t.num_key_value_heads, t.head_dim) == (d.HEADS, d.KV_HEADS, d.HEAD_DIM)
        and (t.linear_num_key_heads, t.linear_num_value_heads) == (d.K_HEADS, d.V_HEADS)
        and (t.linear_key_head_dim, t.linear_value_head_dim) == (d.GDN_DIM, d.GDN_DIM)
        and t.linear_conv_kernel_dim == d.CONV_TAPS
        and getattr(t, "attn_output_gate", False)
        and not getattr(t, "tie_word_embeddings", False)
        and rope.get("partial_rotary_factor") == 0.25
        and float(rope.get("rope_theta", 0)) == d.ROPE_THETA
    )  # fmt: skip


def config_refusal(atom_config) -> str | None:
    """Why this deployment cannot use the Qwen3.8-27B megakernel, or None."""
    hf = atom_config.hf_config
    # Other models answer here, before any field only a full Config has (the GDN
    # pools ask this of every hybrid model, including test stand-ins).
    if not envs.ATOM_MONO_ENABLE:
        return "switched off"
    if not model_matches(hf):
        return "not the Qwen3.8-27B dense hybrid"
    spec = atom_config.speculative_config
    checks = [
        (
            atom_config.tensor_parallel_size == 1,
            f"TP{atom_config.tensor_parallel_size}",
        ),
        (
            getattr(atom_config, "pipeline_parallel_size", 1) == 1
            and getattr(atom_config.parallel_config, "data_parallel_size", 1) == 1,
            "PP / DP",
        ),
        (
            atom_config.kv_cache_dtype == "bf16",
            f"kv cache {atom_config.kv_cache_dtype}",
        ),
        (
            atom_config.kv_cache_block_size % 8 == 0,
            f"KV block size {atom_config.kv_cache_block_size}",
        ),
        (
            atom_config.max_model_len <= MAX_MODEL_LEN,
            f"max_model_len {atom_config.max_model_len} > {MAX_MODEL_LEN}",
        ),
        # the kernel's commit writes ATOM's per-token state slots; ReplaySSM's
        # checkpoint + record buffers are another layout (off by default here)
        (envs.ATOM_ENABLE_REPLAYSSM is not True, "ReplaySSM"),
        (
            spec is None or spec.num_speculative_tokens <= max_spec_tokens(),
            f"{spec and spec.num_speculative_tokens} speculative tokens > {max_spec_tokens()}",
        ),
    ]
    return first_refusal(checks)


def serves(atom_config) -> bool:
    return config_refusal(atom_config) is None
