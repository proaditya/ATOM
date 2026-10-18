# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GDN replay records for accepted-prefix commit (speculative verify), applied to a verify kernel source.

The verify launch writes the GDN state after all M tokens; a speculative round accepts only a
prefix k <= M. Per row t the core already holds what the recurrence applied: decay d_t, the
delta delta_t = beta_t (v_t - (d_t S_{t-1}) k_t) for this CTA's value columns, and the
normalized key k_t. The records store them per (row, value head) in the diagnostics-only `squares`
slot (260 floats: delta[0:128], key[128:256], d[256]); the host then commits any prefix in
closed form: S_k = (prod_{t<k} d_t) S_0 + sum_{t<k} (prod_{t<s<k} d_s) delta_t k_t^T.
~0.8 MB of records per layer per launch (M = 16), no state snapshots.

  python gen_replay_records.py SRC OUT OLD_BUILDER OLD_SYMBOL SUFFIX
"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
old_builder, old_symbol, suffix = sys.argv[3], sys.argv[4], sys.argv[5]


def sub(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:120])
    src = src.replace(old, new)


sub("def " + old_builder + "(", "def build_replay" + suffix + "(")
sub("kernel_name = '" + old_symbol + "'", "kernel_name = 'qwen" + suffix + "_replay'")
sub(
    """                    core_delta0 = gpu.shuffle_idx(core_delta0, core_wave_lane - core_lane, 64)
                    core_delta1 = gpu.shuffle_idx(core_delta1, core_wave_lane - core_lane, 64)
""",
    """                    core_delta0 = gpu.shuffle_idx(core_delta0, core_wave_lane - core_lane, 64)
                    core_delta1 = gpu.shuffle_idx(core_delta1, core_wave_lane - core_lane, 64)
                    # replay record of row core_row for accepted-prefix commit (squares slot).
                    core_rec = core_squares + (core_row * 48 + core_head) * 260
                    if core_lane == 0:
                        fx.ptr_store(core_delta0, core_rec + core_value_base)
                        fx.ptr_store(core_delta1, core_rec + core_value_base + 1)
                    if (core_split == 0) & (core_subgroup == 0):
                        for core_index in range_constexpr(8):
                            fx.ptr_store(core_keys[core_index], core_rec + 128 + core_lane + core_index * 16)
                        if core_lane == 0:
                            fx.ptr_store(core_d, core_rec + 256)
""",
)
out.write_text(src)
print("wrote", out)
