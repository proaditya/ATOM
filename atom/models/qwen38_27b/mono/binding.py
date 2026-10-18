# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The megakernel's buffers and its 65 x 45 pointer table.

``KernelBinding`` owns everything one build of the kernel reads or writes but
the model's state: the converted weights (shared between builds), every
layer's scratch, the replay records, the native control block, and the table
that names them. The cross-layer hand-offs are aliases, as the kernel was
qualified with: layer i's ``down`` / ``residual2`` are layer i + 1's
``hidden`` / ``residual``, the embedding and a zero residual feed layer 0, one
``mixer`` serves every layer, and a DeltaNet layer's ``state_snapshots`` is
its ``final_state`` (the primary, non-diagnostic build).

The state the kernel carries from launch to launch lives in
``SharedBuffers`` (conv history, KV pages and their page map), except each
DeltaNet layer's recurrent state (``initial_state``, read only), whose address
the step writes into the table on the device (``engine.MonoEngine.step``).
"""

import torch

from atom.models.qwen38_27b.mono import dims as d

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "uint8": torch.uint8,
    "int32": torch.int32,
    "int64": torch.int64,
}


class SharedBuffers:
    """The buffers every build shares, one row a DeltaNet (or attention)
    layer so one Triton launch covers all layers: the conv history the next
    launch reads and the one it writes, the final recurrent state, the
    pre-conv projected rows and the GDN replay records (``max_rows`` rows),
    and each attention layer's KV pages with their shared page map."""

    def __init__(self, max_rows: int, max_pages: int, device) -> None:
        n = len(d.DELTA_LAYERS)
        self.max_rows = max_rows
        self.conv_history = torch.zeros(
            n, d.CONV_DIM * d.CONV_TAPS, dtype=torch.bfloat16, device=device
        )
        self.conv_final = torch.zeros_like(self.conv_history)
        self.final_state = torch.zeros(
            n, d.V_HEADS * d.GDN_DIM * d.GDN_DIM, dtype=torch.float32, device=device
        )
        self.raw = torch.zeros(
            n, max_rows * d.PROJECTED, dtype=torch.bfloat16, device=device
        )
        self.records = torch.zeros(
            n,
            max_rows * d.V_HEADS * d.RECORD_FLOATS,
            dtype=torch.float32,
            device=device,
        )
        self.max_pages = max_pages
        self.block_map = torch.arange(max_pages, dtype=torch.int32, device=device)
        self.kv = [
            torch.zeros(
                max_pages * d.PAGE_ELEMENTS, dtype=torch.bfloat16, device=device
            )
            for _ in d.ATTENTION_LAYERS
        ]
        self.kv_ptrs = torch.tensor(
            [t.data_ptr() for t in self.kv], dtype=torch.int64, device=device
        )


def _empty(name: str, nbytes: int, device) -> torch.Tensor:
    dtype = _DTYPES[d.dtype_name(name)]
    count = nbytes // torch.empty((), dtype=dtype).element_size()
    return torch.zeros(count, dtype=dtype, device=device)


# DeltaNet buffers that carry state between launches: ``SharedBuffers``
_SHARED = frozenset(("raw", "conv_final", "final_state"))


class KernelBinding:
    """The buffers of an ``rows``-row build and the table naming them.

    ``weights`` is ``(layers, endpoints)`` from ``weights.convert_model``,
    ``shared`` the state-carrying buffers every build reads and writes; the
    recurrent state each step reads is bound on the device.
    """

    def __init__(self, rows: int, weights, shared: SharedBuffers, device) -> None:
        if rows % 8 or rows > shared.max_rows:
            raise ValueError(f"{rows} rows: the kernel runs 8 or 16")
        self.rows = rows
        self.device = device
        scale = rows // 8
        layer_weights, endpoint_weights = weights

        def scratch(name: str, nbytes: int) -> torch.Tensor:
            return _empty(
                name, nbytes * (scale if name in d.ROW_DEPENDENT else 1), device
            )

        self.endpoints = dict(endpoint_weights)
        for name, nbytes in d.ENDPOINT_SCRATCH_BYTES.items():
            self.endpoints[name] = scratch(name, nbytes)
        self.endpoints["token_ids"] = torch.zeros(
            rows, dtype=torch.int64, device=device
        )
        # the native control block, zeroed with the per-layer controls every
        # launch (``launch_kernels.launch_begin``)
        self.native = torch.zeros(
            d.NATIVE_WORDS + 128, dtype=torch.int32, device=device
        )
        self.controls = torch.zeros(
            d.LAYERS, d.CONTROL_WORDS, dtype=torch.int32, device=device
        )
        self.command = torch.zeros(1, dtype=torch.int32, device=device)
        self.mixer = scratch("mixer", d.SHARED_MIXER_BYTES)
        # attention metadata shared by the 16 attention layers: each row's
        # position and the page its KV is written to, the page map
        self.positions = torch.zeros(rows, dtype=torch.int32, device=device)
        self.context_blocks = torch.zeros(rows, dtype=torch.int32, device=device)
        self.block_map = shared.block_map
        self.tasks = _empty("tasks", d.TASKS_BYTES, device)
        self.dispatch = _empty("dispatch", d.DISPATCH_BYTES, device)

        self.layers: list[dict[str, torch.Tensor]] = []
        previous = None
        for i in range(d.LAYERS):
            buffers = dict(layer_weights[i])
            sizes = (
                d.ATTENTION_SCRATCH_BYTES
                if d.is_attention(i)
                else d.DELTA_SCRATCH_BYTES
            )
            for name, nbytes in sizes.items():
                if name not in _SHARED:
                    buffers[name] = scratch(name, nbytes)
            if previous is None:
                buffers["hidden"] = self.endpoints["embedded"]
                buffers["residual"] = self.endpoints["initial_residual"]
            else:
                buffers["hidden"] = previous["down"]
                buffers["residual"] = previous["residual2"]
            buffers["mixer"] = self.mixer
            buffers["control"] = self.controls[i]
            if d.is_attention(i):
                for name in (
                    "positions",
                    "context_blocks",
                    "block_map",
                    "tasks",
                    "dispatch",
                ):
                    buffers[name] = getattr(self, name)
                kv = shared.kv[d.ATTENTION_LAYERS.index(i)]
                buffers["context"] = buffers["incoming_context"] = kv
            else:
                j = d.DELTA_LAYERS.index(i)
                # the GDN replay records (gen_replay_records) in the diagnostics-only squares
                # slot: [row][value head][delta 128 | key 128 | decay]
                buffers["squares"] = shared.records[
                    j, : rows * d.V_HEADS * d.RECORD_FLOATS
                ]
                buffers["raw"] = shared.raw[j, : rows * d.PROJECTED]
                buffers["conv_history"] = shared.conv_history[j]
                buffers["conv_final"] = shared.conv_final[j]
                buffers["final_state"] = buffers["state_snapshots"] = (
                    shared.final_state[j]
                )
                # re-pointed every step at the state slot the step reads
                buffers["initial_state"] = shared.final_state[j]
            self.layers.append(buffers)
            previous = buffers
        self.endpoints["final_hidden"] = previous["down"]
        self.endpoints["final_skip"] = previous["residual2"]
        self.endpoints["native"] = self.native
        self.table = torch.zeros(
            d.TABLE_ROWS, d.TABLE_CELLS, dtype=torch.int64, device=device
        )
        self.write_table()

    # ---- table ---------------------------------------------------------

    def write_table(self) -> None:
        """Fill the pointer table from the buffers (host, once)."""
        rows = []
        for i, buffers in enumerate(self.layers):
            names = d.ATTENTION_SLOTS if d.is_attention(i) else d.DELTA_SLOTS
            row = [buffers[name].data_ptr() for name in names]
            if d.is_attention(i):
                row.append(buffers["context"].numel())
            else:
                row += [0, 0]
            rows.append(row)
        endpoint_row = [0] * d.TABLE_CELLS
        for slot, name in d.ENDPOINT_SLOTS.items():
            endpoint_row[slot] = self.endpoints[name].data_ptr()
        rows.append(endpoint_row)
        for i, row in enumerate(rows):
            slots = (
                d.TABLE_CELLS if i == d.LAYERS else (44 if d.is_attention(i) else 43)
            )
            if any(p % 16 for p in row[:slots]):
                raise ValueError(
                    f"table row {i}: kernel buffers must be 16-byte aligned"
                )
        self.table.copy_(torch.tensor(rows, dtype=torch.int64))

    @property
    def failed(self) -> torch.Tensor:
        """The kernel's failure word (non-zero: a wait gave up)."""
        return self.native[d.FAILURE_WORD]

    @property
    def logits(self) -> torch.Tensor:
        return self.endpoints["logits_f32"].view(self.rows, d.VOCAB)

    def residual1(self, layer: int) -> torch.Tensor:
        """The residual stream entering layer ``layer`` after its norm's fold
        (= the stream after layer ``layer - 1``), [rows, HIDDEN] bf16."""
        return self.layers[layer]["residual1"].view(self.rows, d.HIDDEN)
