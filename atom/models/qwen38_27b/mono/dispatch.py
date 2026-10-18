# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which Qwen3.8-27B forwards run on the megakernel.

``install_mono_decode`` wraps the loaded model (``model_runner.
mono_decode_installers``); every attribute passes through to it. A forward
runs on the megakernel when it is one sequence's decode or speculative verify
of up to the widest loaded build's rows (8 or 16) -- its whole forward, LM head included,
in one launch (``engine``). Everything else -- prefill, batches of several
sequences -- keeps the per-op model, after two pieces of bookkeeping that
keep the two paths interchangeable at any step:

- a verify whose accepted prefix is not yet committed is settled first, into
  the state slot the per-op step reads (or dropped, if a prefill takes the
  slot over);
- the KV mirror forgets what it holds (``engine.invalidate_mirror``).

The weights are converted and the kernels loaded here, at install, so the
memory they take is accounted before the KV cache is sized.
"""

import logging
import time

import torch
from torch import nn

from atom.models.qwen38_27b.mono import dims as d
from atom.models.qwen38_27b.mono import launch_kernels as lk
from atom.models.qwen38_27b.mono import weights as W
from atom.models.qwen38_27b.mono.config import config_refusal, kernel_names
from atom.models.qwen38_27b.mono.engine import MonoEngine, Pools, StepInputs
from atom.models.qwen38_27b.mono.verdict import RowArgmax, chain_verdict
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

# the verdicts handed to ATOM, a few steps deep: the runner copies them to
# the host on a side stream the allocator does not see, so a fresh tensor
# could be reused under that copy
_VERDICT_RING = 8


class MonoDecodeModel(nn.Module):
    """The loaded Qwen3.5 model, its single-sequence decode steps routed to
    the megakernel."""

    def __init__(
        self, model: nn.Module, atom_config, drafter, metadata_builder
    ) -> None:
        super().__init__()
        self.model = model
        self._builder = metadata_builder
        self._drafter = drafter
        self._block_size = atom_config.kv_cache_block_size
        t0 = time.time()
        get = W.safetensors_source(
            atom_config.model, torch.device("cuda", torch.cuda.current_device())
        )
        weights = W.convert_model(get)
        names = kernel_names()
        self.engine = MonoEngine(weights, atom_config.max_model_len, names)
        # the build serving a launch of n rows: the narrowest that holds them
        self.by_rows = sorted((self.engine.binding(n).rows, n) for n in names)
        self.rows = self.by_rows[-1][0]
        torch.cuda.synchronize()
        logger.info(
            "Qwen3.8-27B megakernel ready (%.1f s): %s",
            time.time() - t0,
            ", ".join(names),
        )
        self._pools = None
        self._logits = None
        self.mono_steps = 0
        dev = torch.device("cuda", torch.cuda.current_device())
        self._words = lk.StepWords(self.rows, dev)
        # a sampled verify's commit inputs (accepted, state slot)
        self._commit_words = torch.zeros(2, dtype=torch.int32, device=dev)
        self._one = torch.ones(1, dtype=torch.int32, device=dev)
        if getattr(drafter, "mono_aux_reader", False):
            drafter.mono_aux_source = self
        # a verify's sampling (``verdict``): the target's token at each row
        self._yv = torch.zeros(self.rows, dtype=torch.int64, device=dev)
        self._argmax = RowArgmax(self.rows, d.VOCAB, dev)
        self._verdicts = {}
        self._ring = 0
        self._bonus = torch.zeros(1, dtype=torch.int64, device=dev)

    # ---- routing ---------------------------------------------------------

    def _eligible(self, q: int, inputs_embeds, fwd) -> bool:
        ctx = fwd.context
        if inputs_embeds is not None or ctx is None or ctx.is_prefill or ctx.is_draft:
            return False
        if fwd.ubatch_slices is not None or ctx.scheduled_bs != 1:
            return False
        gdn = getattr(fwd.attn_metadata, "gdn_metadata", None)
        if gdn is None or gdn.num_prefills or getattr(gdn, "replayssm", False):
            return False
        return 0 < q <= self.rows

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kw
    ):
        fwd = get_forward_context()
        if self._eligible(input_ids.numel(), inputs_embeds, fwd):
            return self._mono_forward(input_ids, positions, fwd)
        self._logits = None
        self._per_op_bookkeeping(fwd)
        if inputs_embeds is None and intermediate_tensors is None:
            return self.model(input_ids, positions)
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kw
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._logits is not None:
            return self._logits[: hidden_states.shape[0]]
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    # ---- the per-op path ---------------------------------------------------

    def _per_op_bookkeeping(self, fwd) -> None:
        if self._pools is None:
            return
        self.engine.invalidate_mirror()
        gdn = getattr(fwd.attn_metadata, "gdn_metadata", None)
        if gdn is None:
            return
        eng = self.engine
        if gdn.num_prefills:
            # a prefill that takes the pending request's slot over drops it
            slots = gdn.non_spec_state_indices_tensor
            if slots is not None:
                hit = (slots.view(-1) == eng.pending_slot0).any()
                eng.pending.mul_((~hit).to(torch.int32))
            return
        spec = gdn.spec_state_indices_tensor
        if spec is None or gdn.num_accepted_tokens is None:
            return
        match = spec[:, 0] == eng.pending_slot0
        found = match.any().to(torch.int32).reshape(1)
        row = match.to(torch.int32).argmax()
        accepted = gdn.num_accepted_tokens[row].to(torch.int32).reshape(1)
        dst = spec[row].gather(0, (accepted - 1).clamp_min(0).long()).to(torch.int32)
        eng.pending.mul_(found)
        eng.commit(self._pools, accepted, dst)

    # ---- the megakernel path -----------------------------------------------

    def _bind_pools(self, fwd) -> Pools:
        runner = self._builder.model_runner
        kv = fwd.kv_cache_data
        ks = [kv[f"layer_{i}"].k_cache for i in d.ATTENTION_LAYERS]
        vs = [kv[f"layer_{i}"].v_cache for i in d.ATTENTION_LAYERS]
        bs = self._block_size
        for k, v in zip(ks, vs):
            if k.dtype != torch.bfloat16 or tuple(k.shape[1:]) != (
                d.KV_HEADS,
                d.HEAD_DIM // 8,
                bs,
                8,
            ):
                raise RuntimeError(
                    f"unexpected K cache view {k.dtype} {tuple(k.shape)}"
                )
            if tuple(v.shape[1:]) != (d.KV_HEADS, bs // 8, d.HEAD_DIM, 8):
                raise RuntimeError(f"unexpected V cache view {tuple(v.shape)}")
        conv, state = runner.mamba_k_cache, runner.mamba_v_cache
        if (
            state.dtype != torch.float32
            or tuple(state.shape[2:]) != (d.V_HEADS, d.GDN_DIM, d.GDN_DIM)
            or state.stride()[2:] != (d.GDN_DIM * d.GDN_DIM, 1, d.GDN_DIM)
        ):
            # ATOM's recurrent kernels index h[k][v]; the megakernel's memory is
            # [v][k], the pool's transposed (VK) layout
            raise RuntimeError(
                f"recurrent state pool {state.dtype} {tuple(state.shape)} {state.stride()}"
            )
        if conv.shape[-1] != d.CONV_DIM or conv.dtype != torch.bfloat16:
            raise RuntimeError(f"conv state pool {conv.dtype} {tuple(conv.shape)}")
        dev = state.device
        return Pools(
            conv=conv, state=state,
            k_ptrs=torch.tensor([t.data_ptr() for t in ks], dtype=torch.int64, device=dev),
            v_ptrs=torch.tensor([t.data_ptr() for t in vs], dtype=torch.int64, device=dev),
            block_size=bs,
        )  # fmt: skip

    def _mono_forward(self, input_ids, positions, fwd) -> torch.Tensor:
        if self._pools is None:
            self._pools = self._bind_pools(fwd)
        md = fwd.attn_metadata
        gdn = md.gdn_metadata
        eng = self.engine
        q = input_ids.numel()
        spec = gdn.spec_state_indices_tensor
        speculative = gdn.num_spec_decodes > 0 and spec is not None
        name = self._build_for(q)
        rows = eng.binding(name).rows
        pos = positions if positions.dim() == 1 else positions[0]
        table = md.block_tables[0]
        w = self._words
        if speculative:
            lk.step_inputs(
                w, input_ids.view(-1), pos, table, rows, self._block_size, spec=spec[0],
                num_accepted=gdn.num_accepted_tokens, pending_slot0=eng.pending_slot0, pending=eng.pending,
            )  # fmt: skip
        else:
            lk.step_inputs(
                w, input_ids.view(-1), pos, table, rows, self._block_size,
                ns_in=gdn.non_spec_state_indices_in_tensor, ns_out=gdn.non_spec_state_indices_tensor,
            )  # fmt: skip
        word = w.word
        inp = StepInputs(
            token_ids=w.token_ids[:rows], base=word(lk.W_BASE), real_rows=word(lk.W_ROWS),
            state_slot=word(lk.W_STATE), conv_in=word(lk.W_CONV_IN), window=word(lk.W_WINDOW),
            conv_out=word(lk.W_CONV_OUT), block_table=table, kv_slots=w.kv_slots[:rows],
            request=word(lk.W_CONV_OUT),
        )  # fmt: skip
        if speculative:
            # settle the previous verify (into the slot this step reads), if
            # it is this request's (``step_inputs`` gated ``pending``)
            eng.commit(
                self._pools, word(lk.W_ACCEPTED), inp.state_slot, arm_follows=True
            )
        b = eng.step(name, self._pools, inp)
        if not speculative:
            # one row, accepted: commit now, into the slot ATOM writes
            eng.commit(self._pools, self._one, inp.conv_out)
        self._write_aux(b, q)
        self._logits = b.logits
        self.mono_steps += 1
        return b.endpoints["normalized"].view(rows, d.HIDDEN)[:q]

    def _build_for(self, q: int) -> str:
        """The build a launch of ``q`` rows runs on: the narrowest holding them."""
        return next(x for x in self.by_rows if x[0] >= q)[1]

    def mono_aux(self, n: int, layers) -> list[torch.Tensor] | None:
        """The residual streams after ``layers`` of this step's ``n`` rows,
        read in place from the kernel's buffers, when this step ran on the
        megakernel (else None). For a drafter that registered itself as
        ``mono_aux_reader``: the step then skips copying its aux into the
        drafter's capture buffers."""
        fwd = get_forward_context()
        if self._pools is None or not self._eligible(n, None, fwd):
            return None
        return self.engine.aux(self.engine.binding(self._build_for(n)), n, layers)

    def verify_on_mono(self, n_target: int, bonus_token_ids) -> bool:
        """Did this step's verify (``n_target`` drafts, one request) run on the
        megakernel?"""
        if bonus_token_ids is not None and bonus_token_ids.numel() != 1:
            return False
        return self._pools is not None and self._eligible(
            n_target + 1, None, get_forward_context()
        )

    def verdict(self, n_target: int):
        """The rejection sampler's result for a greedy chain verify that ran on
        the megakernel: the argmax of each row of the kernel's logits, the
        drafts accepted while they match it, as ATOM's verdict."""
        n = n_target + 1
        b = self.engine.binding(self._build_for(n))
        yv = self._yv[: b.rows]
        self._argmax(b.logits, n, yv)
        if n not in self._verdicts:
            dev = yv.device
            self._verdicts[n] = (
                torch.zeros(_VERDICT_RING, n, dtype=torch.int32, device=dev),
                torch.zeros(_VERDICT_RING, 3, 1, dtype=torch.int32, device=dev),
            )
        outs, verdicts = self._verdicts[n]
        i = self._ring = (self._ring + 1) % _VERDICT_RING
        out, verdict = outs[i], verdicts[i]
        chain_verdict(yv, self._words.token_ids[: b.rows], out, verdict, n - 1)
        return out.view(1, -1), verdict.view(3, 1)

    def commit_verify(self, metadata, last_token_indices) -> None:
        """After sampling a verify: commit the pending launch's accepted prefix
        (rows 0 .. last_token_indices[0]) into the slot the next step reads.
        A no-op on the device unless a megakernel verify is pending."""
        gdn = getattr(metadata, "gdn_metadata", None)
        if self._pools is None or gdn is None or gdn.spec_state_indices_tensor is None:
            return
        eng = self.engine
        w = self._commit_words
        lk.commit_inputs(
            w,
            last_token_indices,
            gdn.spec_state_indices_tensor[0],
            eng.pending_slot0,
            eng.pending,
        )
        eng.commit(self._pools, w[:1], w[1:2])

    def _write_aux(self, b, q: int) -> None:
        """The drafter's aux hidden states, which the per-op model emits through
        its forward hook, from the kernel's residual streams."""
        buffers = getattr(self._drafter, "_aux_buffers", None)
        if not buffers or getattr(self._drafter, "mono_aux_reader", False):
            return
        for buf, aux in zip(
            buffers, self.engine.aux(b, q, self._drafter.aux_layer_ids), strict=True
        ):
            buf[:q].copy_(aux)


def install_mono_decode(
    model: nn.Module, atom_config, drafter, metadata_builder
) -> nn.Module:
    why = config_refusal(atom_config)
    if why is not None:
        logger.info("Qwen3.8-27B mono decode off: %s", why)
        return model
    wrapped = MonoDecodeModel(model, atom_config, drafter, metadata_builder)
    # A verify's accepted prefix is known once it is sampled: the runner hands
    # it to the builder before the draft reads anything, and the recurrent
    # state is committed there, in the same step.
    commit = metadata_builder.commit_speculative_state

    def commit_speculative_state(metadata, last_token_indices):
        wrapped.commit_verify(metadata, last_token_indices)
        return commit(metadata, last_token_indices)

    metadata_builder.commit_speculative_state = commit_speculative_state
    runner = metadata_builder.model_runner
    rejection = getattr(runner, "rejection_sampler", None)
    if rejection is not None:
        _route_sampling(wrapped, runner.sampler, rejection)
    return wrapped


def _route_sampling(wrapped: MonoDecodeModel, sampler, rejection) -> None:
    """A greedy chain verify that ran on the megakernel is sampled by
    ``wrapped.verdict`` (two argmax launches over the kernel's logits and the
    accept walk, against a dozen generic kernels) when the generic sampler
    would also take the target's argmax (no synthetic or relaxed acceptance).
    The bonus row's own sampling is then skipped: the verdict reads every
    row."""
    from atom.model_ops import rejection_sampler as rs

    def fused_greedy(n_target: int, bonus, spec) -> bool:
        # the verdict's width is the whole chain's (``num_spec_steps``)
        return (
            rejection.synthetic_acceptance_rates is None
            and rs.RELAXED_TOP_N <= 1
            and n_target == spec.num_spec_steps
            and wrapped.verify_on_mono(n_target, bonus)
        )

    chain = rejection.forward

    def forward(metadata, target_logits, bonus_token_ids, *, target_token_ids=None):
        n = target_logits.shape[0]
        if target_token_ids is None and fused_greedy(n, bonus_token_ids, metadata):
            return wrapped.verdict(n)
        return chain(
            metadata, target_logits, bonus_token_ids, target_token_ids=target_token_ids
        )

    rejection.forward = forward
    sample = sampler.forward

    def sample_bonus(
        logits,
        temperatures,
        top_ks=None,
        top_ps=None,
        all_greedy=False,
        needs_independent_noise=False,
    ):
        spec = get_forward_context().spec_decode_metadata
        if (
            all_greedy
            and spec is not None
            and logits.shape[0] == 1
            and fused_greedy(spec.target_logits_indices.numel(), None, spec)
        ):
            return (
                wrapped._bonus
            )  # unread: the verdict takes the bonus row's argmax itself
        return sample(
            logits, temperatures, top_ks, top_ps, all_greedy, needs_independent_noise
        )

    sampler.forward = sample_bonus
