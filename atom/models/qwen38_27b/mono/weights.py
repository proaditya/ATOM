# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The megakernel's weights, converted from the Hugging Face checkpoint.

Every buffer a layer row of the kernel's pointer table names as an input is
built here, in the kernel's layout and byte for byte what the reference
compile produced for the checkpoints the kernel was qualified on
(``check_native`` compares each against its recorded SHA-256):

- Gemma-style RMSNorm weights are stored shifted (``w + 1``), except the
  DeltaNet output norm;
- every projection is MXFP4 (``mxfp4``): the DeltaNet ``q | k | v | z | b | a``
  rows as one ``combined`` matrix of 16512 rows (``b`` and ``a`` quantized by
  the CPU rule, 32 zero rows of padding), the full-attention ``q+gate | k | v``
  as one, and gate / up interleaved row by row;
- each matrix's scales in the tile layout its kernel stage reads (small,
  large64 or large256).

The source tensors are read straight from the safetensors files, not from
the per-op model's parameters, whose fused layouts differ.
"""

import hashlib
import json
import logging
import os
from collections.abc import Callable

import torch

from atom.models.qwen38_27b.mono import mxfp4 as mx
from atom.models.qwen38_27b.mono.dims import (
    ATTENTION_INPUT_BYTES,
    DELTA_INPUT_BYTES,
    ENDPOINT_INPUT_BYTES,
    LAYERS,
    is_attention,
)

logger = logging.getLogger("atom")

PREFIX = "model.language_model."
# rows quantized at a time: bounds the fp32 intermediates of the LM head
_ROW_CHUNK = 8192

Source = Callable[[str], torch.Tensor]


def safetensors_source(model_dir: str, device) -> Source:
    """A reader of the checkpoint's tensors by name, onto ``device``."""
    from safetensors import safe_open

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            files = json.load(f)["weight_map"]
    else:
        single = os.path.join(model_dir, "model.safetensors")
        with safe_open(single, "pt") as f:
            files = {name: "model.safetensors" for name in f}
    handles = {}

    def get(name: str) -> torch.Tensor:
        file = files[name]
        if file not in handles:
            handles[file] = safe_open(os.path.join(model_dir, file), "pt")
        return handles[file].get_tensor(name).to(device)

    return get


def _quantized(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(packed [N, K/2], canonical scales [N, K/32]), quantized by row chunks."""
    parts = [
        mx.quantize(weight[i : i + _ROW_CHUNK])
        for i in range(0, weight.shape[0], _ROW_CHUNK)
    ]
    return torch.cat([p for p, _ in parts]), torch.cat([s for _, s in parts])


def _dense(
    weight: torch.Tensor, scale_layout: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """A projection's (preshuffled weight, scales in ``scale_layout``)."""
    packed, scales = _quantized(weight.to(torch.bfloat16).contiguous())
    return mx.preshuffle_weight(packed), _scales(scales, scale_layout)


def _scales(scales: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "small":
        return mx.preshuffle_scales_small(scales)
    return mx.pack_scales(scales, int(layout.removeprefix("large")))


def _shifted(norm: torch.Tensor) -> torch.Tensor:
    """Qwen3.5's zero-centred RMSNorm weight as the kernel applies it."""
    return norm + 1


def _combined(get: Source, p: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The DeltaNet input projection: q | k | v | z (GPU rule) then b | a | 0
    (CPU rule), one matrix; large64 scales."""
    qkv, qkv_s = _quantized(get(p + "in_proj_qkv.weight").to(torch.bfloat16))
    z, z_s = _quantized(get(p + "in_proj_z.weight").to(torch.bfloat16))
    b, a = get(p + "in_proj_b.weight"), get(p + "in_proj_a.weight")
    ba = torch.cat((b, a, torch.zeros(32, b.shape[1], dtype=b.dtype, device=b.device)))
    ba_w, ba_s = mx.quantize_nearest_first(ba.to(torch.bfloat16).float())
    weight = torch.cat([mx.preshuffle_weight(w) for w in (qkv, z, ba_w)])
    scales = mx.pack_scales(torch.cat((qkv_s, z_s, ba_s)), 64)
    return weight, scales


def _mlp(get: Source, p: str) -> dict:
    gate, up = get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")
    # rows gate_0, up_0, gate_1, up_1, ...
    interleaved = torch.stack((gate, up), 1).reshape(-1, up.shape[1])
    up_w, up_s = _dense(interleaved, "large256")
    del interleaved
    down_w, down_s = _dense(get(p + "mlp.down_proj.weight"), "large64")
    return {
        "up_weight": up_w,
        "up_scale": up_s,
        "down_weight": down_w,
        "down_scale": down_s,
    }


def convert_layer(get: Source, layer: int) -> dict[str, torch.Tensor]:
    """Layer ``layer``'s kernel inputs that are weights (no state, no KV)."""
    p = f"{PREFIX}layers.{layer}."
    out = {
        "input_norm": _shifted(get(p + "input_layernorm.weight")),
        "post_norm": _shifted(get(p + "post_attention_layernorm.weight")),
    }
    if is_attention(layer):
        a = p + "self_attn."
        qgkv = torch.cat([get(a + f"{x}_proj.weight") for x in "qkv"])
        out["qgkv_weight"], out["qgkv_scale"] = _dense(qgkv, "large64")
        del qgkv
        out["qgamma"] = _shifted(get(a + "q_norm.weight"))
        out["kgamma"] = _shifted(get(a + "k_norm.weight"))
        out["output_weight"], out["output_scale"] = _dense(
            get(a + "o_proj.weight"), "small"
        )
    else:
        d = p + "linear_attn."
        out["combined_weight"], out["combined_scale"] = _combined(get, d)
        out["conv_weight"] = get(d + "conv1d.weight").to(torch.bfloat16).contiguous()
        out["A_log"] = get(d + "A_log").float().contiguous()
        out["dt_bias"] = get(d + "dt_bias").float().contiguous()
        out["head_norm"] = get(d + "norm.weight").to(torch.bfloat16).contiguous()
        out["output_weight"], out["output_scale"] = _dense(
            get(d + "out_proj.weight"), "small"
        )
    out.update(_mlp(get, p))
    expected = ATTENTION_INPUT_BYTES if is_attention(layer) else DELTA_INPUT_BYTES
    for name, value in out.items():
        if value.numel() * value.element_size() != expected[name]:
            raise ValueError(
                f"layer {layer} {name}: {value.numel() * value.element_size()} bytes, kernel reads {expected[name]}"
            )
        out[name] = value.contiguous().view(-1)
    return out


def convert_endpoints(get: Source) -> dict[str, torch.Tensor]:
    """Embedding, final norm and LM head."""
    head_w, head_s = _dense(get("lm_head.weight"), "large64")
    out = {
        "embedding_weight": get(PREFIX + "embed_tokens.weight")
        .to(torch.bfloat16)
        .contiguous(),
        "final_norm": _shifted(get(PREFIX + "norm.weight")).to(torch.bfloat16),
        "head_weight": head_w,
        "head_scale": head_s,
    }
    for name, value in out.items():
        if value.numel() * value.element_size() != ENDPOINT_INPUT_BYTES[name]:
            raise ValueError(f"{name}: {value.numel() * value.element_size()} bytes")
        out[name] = value.contiguous().view(-1)
    return out


def _sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def check_native(layers: list[dict], endpoints: dict, reference: dict) -> list[str]:
    """Every converted tensor whose SHA-256 differs from ``reference`` (the
    recorded reference compile)."""
    bad = []
    for i, (got, want) in enumerate(zip(layers, reference["layers"])):
        for name, digest in want.items():
            if _sha256(got[name]) != digest:
                bad.append(f"layer {i} {name}")
    for name in ("embedding_weight", "final_norm", "head_weight", "head_scale"):
        if _sha256(endpoints[name]) != reference[name]:
            bad.append(name)
    return bad


def convert_model(get: Source) -> tuple[list[dict], dict]:
    layers = []
    for i in range(LAYERS):
        layers.append(convert_layer(get, i))
    return layers, convert_endpoints(get)
