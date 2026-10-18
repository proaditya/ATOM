# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One verify / decode step of the megakernel over ATOM's state pools.

``MonoEngine`` owns the converted weights, one ``KernelBinding`` and code
object a build, and the state the builds share. A step (``step``) runs a
fixed sequence of device operations -- graph-capturable, every size and slot
read from device tensors:

1. ``commit``: settle the previous launch, if it is still pending, into the
   recurrent-state slot ATOM reads next (its accepted prefix, replayed from
   the GDN records);
2. point the table's ``initial_state`` cells at the state slot this step reads;
   gather the conv window into the kernel's history;
3. import into the KV mirror whatever ATOM's cache holds that the mirror
   does not (all of it after any step the mirror did not see);
4. launch over ``rows`` consecutive positions;
5. write back the conv state ATOM's rolling buffer expects and this
   launch's rows' K / V into ATOM's cache; mark the launch pending.

Steps 2-4's bookkeeping and step 5's arming are one Triton launch each
(``launch_kernels``), not dozens of small torch ops in the step's graph.

The KV mirror holds one sequence's pages in the kernel's layout; ATOM's cache
stays the record (every step writes its rows back), so the per-op path can
take over at any step.
"""

import logging
from dataclasses import dataclass

import torch

from atom.models.qwen38_27b.mono import dims as d
from atom.models.qwen38_27b.mono import launch_kernels as lk
from atom.models.qwen38_27b.mono import state_kernels as sk
from atom.models.qwen38_27b.mono.binding import KernelBinding, SharedBuffers
from atom.models.qwen38_27b.mono.kernels.build import VARIANTS, build
from atom.models.qwen38_27b.mono.kernels.hip import Megakernel

logger = logging.getLogger("atom")


@dataclass
class Pools:
    """ATOM's pools the steps read and write."""

    conv: torch.Tensor  # [48, slots, state_len, 10240] bf16
    state: torch.Tensor  # [48, slots, 48, 128, 128] fp32
    k_ptrs: torch.Tensor  # int64 [16]: each attention layer's K view
    v_ptrs: torch.Tensor  # int64 [16]
    block_size: int


@dataclass
class StepInputs:
    """One step's device inputs (int32 [1] tensors unless stated)."""

    token_ids: torch.Tensor  # int64 [rows], padded to the build's rows
    base: torch.Tensor  # position of row 0
    real_rows: torch.Tensor  # rows ATOM scheduled (the rest are padding)
    state_slot: torch.Tensor  # the recurrent state this step reads
    conv_in: torch.Tensor  # conv slot read, its window at window - 1
    window: torch.Tensor
    conv_out: torch.Tensor  # conv slot written
    block_table: torch.Tensor  # int32 [max_blocks]: the sequence's blocks
    kv_slots: torch.Tensor  # int32 / int64 [rows]: each row's cache slot
    # the request's own first state slot, which names it while it runs (the
    # slot this step reads changes with the accepted count)
    request: torch.Tensor


class MonoEngine:
    def __init__(
        self, weights, max_model_len: int, variants=("m8",), device=None
    ) -> None:
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.weights = weights
        max_rows = max(VARIANTS[v].rows for v in variants)
        # positions up to max_model_len plus a launch's padding rows
        self.max_pages = (max_model_len + max_rows + d.PAGE_TOKENS - 1) // d.PAGE_TOKENS
        self.shared = SharedBuffers(max_rows, self.max_pages, self.device)
        self.builds = {}
        for name in variants:
            kernel = Megakernel(build(name))
            binding = KernelBinding(kernel.rows, weights, self.shared, self.device)
            self.builds[name] = (kernel, binding)
        dev = self.device

        def scalar(value=0):
            return torch.full((1,), value, dtype=torch.int32, device=dev)

        # the launch awaiting its commit: whether, and from which state slot
        self.pending = scalar()
        self.pending_src = scalar()
        # the pending request: its first state slot, which names it
        self.pending_slot0 = scalar(-1)
        # the rows a commit replays, in order
        self.identity = torch.arange(max_rows, dtype=torch.int64, device=dev)
        self._ones = scalar(1)
        self._delta_rows = torch.tensor(d.DELTA_LAYERS, dtype=torch.int64, device=dev)
        self._state_col = d.DELTA_SLOTS.index("initial_state")
        # the KV mirror: which state slot's sequence it holds, and how far
        self.mirror_slot = scalar(-1)
        self.mirror_len = scalar()
        self.import_start = scalar()
        self.import_end = scalar()
        self.last = None

    def binding(self, name: str) -> KernelBinding:
        return self.builds[name][1]

    # ---- the step -------------------------------------------------------

    def commit(
        self,
        pools: Pools,
        accepted: torch.Tensor,
        dst_slot: torch.Tensor,
        arm_follows: bool = False,
    ) -> None:
        """Settle the pending launch: the state after its first ``accepted``
        rows into ``dst_slot``. No-op on the device when nothing is pending.
        ``arm_follows``: a launch re-arms ``pending`` next (``step``)."""
        s = self.shared
        sk.state_commit(
            pools.state,
            self.pending_src,
            dst_slot,
            s.records,
            self.identity,
            accepted,
            self.pending,
            s.max_rows,
        )
        if not arm_follows:
            self.pending.zero_()

    def invalidate_mirror(self) -> None:
        """A step the mirror did not see may have changed the sequence."""
        self.mirror_len.zero_()

    def step(self, name: str, pools: Pools, inp: StepInputs) -> KernelBinding:
        """One launch of build ``name`` over the step's rows."""
        kernel, b = self.builds[name]
        s = self.shared
        # 2-4. state slot into the table, the mirror's import range, launch inputs
        lk.launch_begin(
            b, self._delta_rows, self._state_col, pools, inp,
            self.mirror_slot, self.mirror_len, self.import_start, self.import_end,
        )  # fmt: skip
        sk.conv_gather(pools.conv, inp.conv_in, inp.window, s.conv_history)
        # KV mirror: a different sequence, or positions it never saw
        sk.kv_import(
            pools.k_ptrs,
            pools.v_ptrs,
            inp.block_table,
            self.import_start,
            self.import_end,
            s.kv_ptrs,
            pools.block_size,
        )
        kernel.launch(b.table)
        # 5. write-backs
        sk.conv_writeback(
            pools.conv,
            inp.conv_out,
            s.conv_history,
            s.raw,
            inp.real_rows,
            self.identity,
            self._ones,
            self._ones,
        )
        sk.kv_export(
            pools.k_ptrs,
            pools.v_ptrs,
            inp.kv_slots,
            inp.base,
            inp.real_rows,
            s.kv_ptrs,
            b.rows,
            pools.block_size,
        )
        lk.launch_arm(self, inp)
        self.last = b
        return b

    def aux(self, b: KernelBinding, rows: int, layers) -> list[torch.Tensor]:
        """A drafter's aux hidden states (the residual stream after each of
        ``layers``, read where the next layer folds it: ``residual1``) of this
        launch's first ``rows`` rows."""
        return [b.residual1(layer + 1)[:rows] for layer in layers]
