# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The shapes the Qwen3.8-27B megakernel is built for, and its pointer table.

One launch runs the whole model -- embedding, 64 layers (48 Gated DeltaNet,
16 full attention, D/D/D/A repeated), final norm and LM head -- for ``rows``
consecutive positions of one sequence (8 rows, or 16 for the 16-row build). It
reads every buffer through a 65 x 45 table of 64-bit addresses: one row a
layer (``DELTA_SLOTS`` / ``ATTENTION_SLOTS``) and an endpoint row
(``ENDPOINT_SLOTS``). The names and byte sizes below are the kernel's
contract; a row-dependent buffer
holds ``rows / 8`` times its 8-row size.
"""

LAYERS = 64
HIDDEN = 5120
VOCAB = 248320
INTERMEDIATE = 17408
# full attention: 24 query heads (each with its output gate), 4 KV heads of 256
HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
# rotary dims of a head (partial_rotary_factor 0.25) and their base
ROTARY_DIM, ROPE_THETA = 64, 10_000_000.0
# Gated DeltaNet: 16 key heads, 48 value heads of 128; the depthwise conv runs
# over q | k | v (10240 channels) with 4 taps
K_HEADS, V_HEADS, GDN_DIM = 16, 48, 128
CONV_DIM, CONV_TAPS = 2 * K_HEADS * GDN_DIM + V_HEADS * GDN_DIM, 4
# one projected row of a DeltaNet layer: q | k | v | z | b | a | pad
PROJECTED = CONV_DIM + V_HEADS * GDN_DIM + 2 * V_HEADS + 32
# the KV cache: pages of 32 positions, [page][kv head][K | V][32][256] bf16
PAGE_TOKENS = 32
PAGE_ELEMENTS = KV_HEADS * 2 * PAGE_TOKENS * HEAD_DIM
# the GDN replay record of one (row, value head), fp32 (gen_replay_records): the delta
# applied to this head's 128 value columns, the normalized key, the decay
RECORD_FLOATS = 260
RECORD_KEY, RECORD_DECAY = 128, 256

TABLE_ROWS, TABLE_CELLS = LAYERS + 1, 45
# the native control block: 3728 words, a 4096-word counter tail, and (16-row build)
# rows 8-15's peer records past it
NATIVE_WORDS = 3728 + 4096 + 4096
COMMAND_WORD = 288
FAILURE_WORD = 289
MAX_COMMAND_EPOCH = 1_048_575
CONTROL_WORDS = 360

DELTA_SLOTS = (
    "hidden", "residual", "input_norm", "post_norm", "combined_weight",
    "combined_scale", "conv_weight", "conv_history", "initial_state", "A_log",
    "dt_bias", "head_norm", "output_weight", "output_scale", "up_weight",
    "up_scale", "down_weight", "down_scale", "input_normalized", "residual1",
    "input_A", "input_AS", "raw", "workspace", "conv_final", "BA", "core",
    "final_state", "squares", "recurrent", "recurrent_A", "recurrent_AS",
    "mixer", "post_normalized", "residual2", "post_A", "post_AS", "activated",
    "activated_AS", "partial", "down", "state_snapshots", "control",
)  # fmt: skip
ATTENTION_SLOTS = (
    "hidden", "residual", "input_norm", "post_norm", "qgkv_weight",
    "qgkv_scale", "qgamma", "kgamma", "positions", "context_blocks",
    "block_map", "tasks", "output_weight", "output_scale", "up_weight",
    "up_scale", "down_weight", "down_scale", "input_normalized", "residual1",
    "input_A", "input_AS", "qgkv", "query", "gate", "context", "partial",
    "maximum", "denominator", "gated_diagnostic", "attention_A",
    "attention_AS", "mixer", "post_normalized", "residual2", "post_A",
    "post_AS", "activated", "activated_AS", "down_partial", "down",
    "dispatch", "control", "incoming_context",
)  # fmt: skip
ENDPOINT_SLOTS = {
    0: "final_hidden", 1: "final_skip", 2: "final_norm", 4: "head_weight",
    5: "head_scale", 18: "normalized", 19: "final_residual", 20: "A", 21: "AS",
    22: "head_bf16", 32: "logits_f32", 42: "native", 43: "embedding_weight",
    44: "token_ids",
}  # fmt: skip

DELTA_INPUT_BYTES = {
    "input_norm": 10240, "post_norm": 10240, "combined_weight": 42270720,
    "combined_scale": 2641920, "conv_weight": 81920, "A_log": 192,
    "dt_bias": 192, "head_norm": 256, "output_weight": 15728640,
    "output_scale": 983040, "up_weight": 89128960, "up_scale": 5570560,
    "down_weight": 44564480, "down_scale": 2785280,
}  # fmt: skip
ATTENTION_INPUT_BYTES = {
    "input_norm": 10240, "post_norm": 10240, "qgamma": 512, "kgamma": 512,
    "qgkv_weight": 36700160, "qgkv_scale": 2293760, "output_weight": 15728640,
    "output_scale": 983040, "up_weight": 89128960, "up_scale": 5570560,
    "down_weight": 44564480, "down_scale": 2785280,
}  # fmt: skip
ENDPOINT_INPUT_BYTES = {
    "embedding_weight": VOCAB * HIDDEN * 2, "final_norm": HIDDEN * 2,
    "head_weight": VOCAB * HIDDEN // 2, "head_scale": VOCAB * HIDDEN // 32,
}  # fmt: skip

# scratch a layer writes, 8-row bytes (dtype by name, ``DTYPES``)
DELTA_SCRATCH_BYTES = {
    "input_normalized": 81920, "residual1": 81920, "input_A": 40960,
    "input_AS": 2752, "raw": 264192, "workspace": 264192, "conv_final": 81920,
    "BA": 4096, "core": 196608, "final_state": 3145728, "recurrent": 98304,
    "recurrent_A": 49152, "recurrent_AS": 3264, "post_normalized": 81920,
    "residual2": 81920, "post_A": 40960, "post_AS": 2752, "activated": 139264,
    "activated_AS": 8896, "partial": 491520, "down": 81920,
}  # fmt: skip
ATTENTION_SCRATCH_BYTES = {
    "input_normalized": 81920, "residual1": 81920, "input_A": 40960,
    "input_AS": 2752, "qgkv": 229376, "query": 98304, "gate": 98304,
    "partial": 50331648, "maximum": 393216, "denominator": 393216,
    "gated_diagnostic": 98304, "attention_A": 49152, "attention_AS": 3264,
    "post_normalized": 81920, "residual2": 81920, "post_A": 40960,
    "post_AS": 2752, "activated": 139264, "activated_AS": 8896,
    "down_partial": 491520, "down": 81920,
}  # fmt: skip
ENDPOINT_SCRATCH_BYTES = {
    "embedded": 81920, "initial_residual": 81920, "normalized": 81920,
    "final_residual": 81920, "A": 40960, "AS": 2752,
    "head_bf16": 3973120, "logits_f32": 7946240,
}  # fmt: skip
SHARED_MIXER_BYTES = 81920
# attention metadata read-only to the kernel at these sizes; the 8-row build derives its
# split layout from the position and never reads the task tables
TASKS_BYTES, DISPATCH_BYTES = 8704, 16

# buffers sized by the launch's rows
ROW_DEPENDENT = frozenset((
    "hidden", "residual", "input_normalized", "residual1", "input_A", "raw",
    "workspace", "BA", "core", "squares", "recurrent", "recurrent_A", "mixer",
    "post_normalized", "residual2", "post_A", "activated", "partial", "down",
    "qgkv", "query", "gate", "maximum", "denominator", "gated_diagnostic",
    "attention_A", "down_partial", "positions", "context_blocks",
    "embedded", "initial_residual", "normalized", "final_residual", "A",
    "head_bf16", "logits_f32", "token_ids",
))  # fmt: skip

UINT8 = frozenset((
    "combined_weight", "combined_scale", "qgkv_weight", "qgkv_scale",
    "output_weight", "output_scale", "up_weight", "up_scale", "down_weight",
    "down_scale", "head_weight", "head_scale", "input_A", "input_AS",
    "recurrent_A", "recurrent_AS", "attention_A", "attention_AS", "post_A",
    "post_AS", "activated", "activated_AS", "A", "AS",
))  # fmt: skip
FLOAT32 = frozenset((
    "A_log", "dt_bias", "initial_state", "final_state", "BA", "core",
    "squares", "partial", "down_partial", "maximum", "denominator",
    "logits_f32",
))  # fmt: skip
INT32 = frozenset(
    ("positions", "context_blocks", "block_map", "tasks", "dispatch", "control")
)


def is_attention(layer: int) -> bool:
    return layer % 4 == 3


ATTENTION_LAYERS = tuple(i for i in range(LAYERS) if is_attention(i))
DELTA_LAYERS = tuple(i for i in range(LAYERS) if not is_attention(i))


def dtype_name(name: str) -> str:
    """The element type of the buffer ``name``."""
    if name == "token_ids":
        return "int64"
    if name in UINT8:
        return "uint8"
    if name in FLOAT32:
        return "float32"
    if name in INT32:
        return "int32"
    return "bfloat16"
