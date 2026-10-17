# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""EAGLE-4 draft model (``LlamaForCausalLMEagle4``), e.g.
EAGLE-Corp/EAGLE4-Qwen3.8-27B.

The EAGLE 3.1 layer (``Eagle3LlamaModel``: one decoder layer over
``concat(norm(embed), norm(fc(norm_i(aux_i))))``, ``norm_output``, a sliding
window) with a low-rank candidate head in front of the LM head: a token's
128 candidates are the top scores of ``(h @ down^T) @ eff^T`` (rank 128), and
the drafted token is the best of their exact logits. The checkpoint names the
layer ``layers.0`` and adds

    efficient_lm_head_down_proj.weight   [rank, hidden]
    efficient_lm_head.weight             [vocab, rank]

The aux hidden states are the target's residual stream AFTER layers
``eagle_aux_hidden_state_layer_ids`` (``Eagle4Proposer`` taps the layer
outputs).
"""

import torch
from torch import nn

from atom.config import Config
from atom.model_ops.utils import atom_parameter
from atom.models.eagle3_llama import Eagle3LlamaModel


class Eagle4LlamaModel(Eagle3LlamaModel):
    def __init__(self, atom_config: Config, prefix: str = "", layer_offset: int = 0):
        config = atom_config.hf_config
        rope = getattr(config, "rope_parameters", None) or {}
        if "rope_theta" in rope and getattr(config, "rope_theta", None) is None:
            config.rope_theta = rope["rope_theta"]
        super().__init__(atom_config, prefix=prefix, layer_offset=layer_offset)
        head = getattr(config, "efficient_lm_head_config", None) or {}
        self.num_candidates = head.get("num_candidates", 128)
        rank = head.get("rank", 128)
        self.efficient_lm_head_down_proj = nn.Module()
        self.efficient_lm_head_down_proj.weight = atom_parameter(
            torch.empty(rank, config.hidden_size, dtype=torch.bfloat16)
        )
        self.efficient_lm_head = nn.Module()
        self.efficient_lm_head.weight = atom_parameter(
            torch.empty(config.vocab_size, rank, dtype=torch.bfloat16)
        )

    def candidates(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Each row's ``num_candidates`` candidate tokens (unordered)."""
        low = hidden_states @ self.efficient_lm_head_down_proj.weight.t()
        scores = low @ self.efficient_lm_head.weight.t()
        return scores.topk(self.num_candidates, dim=-1, sorted=False).indices

    def _whole_lm_head(self) -> torch.Tensor:
        """The LM head over the whole vocabulary: candidates index all of it,
        so a vocab-parallel (TP > 1) head is gathered once, at the first draft
        (the eager warmup)."""
        w = self.lm_head.weight
        vocab = self.efficient_lm_head.weight.shape[0]
        if w.shape[0] >= vocab:
            return w
        whole = getattr(self, "_lm_whole", None)
        if whole is None:
            from aiter.dist.parallel_state import get_tp_group

            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "the EAGLE-4 LM head gather must run before graph capture"
                )
            whole = self._lm_whole = (
                get_tp_group().all_gather(w.contiguous(), dim=0)[:vocab].contiguous()
            )
        return whole

    def compute_draft_ids(
        self, hidden_states: torch.Tensor, *, out: torch.Tensor
    ) -> torch.Tensor:
        """Greedy draft tokens: the best exact logit among the candidates
        (hidden_states is the normed output, ``norm_output``)."""
        cand = self.candidates(hidden_states)
        rows = self._whole_lm_head()[cand]  # [n, C, hidden]
        logits = torch.einsum("nh,nch->nc", hidden_states.float(), rows.float())
        best = cand.gather(1, logits.argmax(1, keepdim=True)).squeeze(1)
        return out.copy_(best.to(out.dtype))
