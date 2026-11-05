"""Copy a GGUF with only its first N decoder layers, optionally with our head baked in.

llama.cpp refuses `--override-kv <arch>.block_count` on a full file ("wrong number of
tensors"), so the early exit has to be a file whose tensors stop at layer N - 1 and whose
block_count says N. The final norm and the embeddings are kept, so `/embeddings` with
pooling last returns the normed state after layer N - 1, which is what a head at that layer
reads. Mirrors gguf_new_metadata.py's copy, with a tensor filter.

With `--head head.npz` (the product layout export_head.py writes: coef, intercept,
scaler_mean, scaler_scale, labels) the standardisation and the logistic weights fold into
one linear layer, written as the `cls.output` tensors that llama.cpp's Qwen3.5 graph applies
to the post-norm state when `<arch>.embedding_length_out` names its size
(`src/models/qwen35.cpp`, create_tensor of LLM_TENSOR_CLS_OUT and the `if (model.cls_out)`
block after `result_norm`; read 2026-11-05). The server's `/embeddings` with pooling last
then returns one logit per label instead of the 2,560-number state.

    uv run python truncate_gguf.py hf/gguf/Qwen3.5-4B-Q4_K_M.gguf hf/gguf/Qwen3.5-4B-Q4_K_M-L8.gguf 8
    uv run python truncate_gguf.py src.gguf dst.gguf 32 --head runs/latency/head_llama32.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gguf
import numpy as np


def folded_head(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """logits = ((x - mean) / scale) @ coef.T + intercept == x @ W.T + b."""
    h = np.load(path)
    coef, intercept = h["coef"].astype(np.float64), h["intercept"].astype(np.float64)
    mean, scale = (
        h["scaler_mean"].astype(np.float64),
        h["scaler_scale"].astype(np.float64),
    )
    weight = coef / scale
    bias = intercept - (mean / scale) @ coef.T
    labels = json.loads(str(h["labels"]))
    x = np.random.default_rng(0).standard_normal(coef.shape[1])
    want = ((x - mean) / scale) @ coef.T + intercept
    assert np.allclose(x @ weight.T + bias, want, atol=1e-6)
    return weight.astype(np.float32), bias.astype(np.float32), labels


def main(src: Path, dst: Path, layers: int, head: Path | None) -> None:
    reader = gguf.GGUFReader(src)
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(dst, arch)
    extra: list[tuple[str, np.ndarray]] = []
    if head is not None:
        weight, bias, labels = folded_head(head)
        extra = [("cls.output.weight", weight), ("cls.output.bias", bias)]
        print(f"head: {len(labels)} labels folded into cls.output {weight.shape}")
    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = layers if field.name == f"{arch}.block_count" else field.contents()
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)
    if extra:
        writer.add_uint32(f"{arch}.embedding_length_out", extra[0][1].shape[0])
    keep = [
        t
        for t in reader.tensors
        if not t.name.startswith("blk.") or int(t.name.split(".")[1]) < layers
    ]
    for t in keep:
        writer.add_tensor_info(
            t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type
        )
    for name, arr in extra:
        writer.add_tensor_info(
            name, arr.shape, arr.dtype, arr.nbytes, gguf.GGMLQuantizationType.F32
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in keep:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    for _, arr in extra:
        writer.write_tensor_data(arr)
    writer.close()
    check = gguf.GGUFReader(dst)
    assert check.fields[f"{arch}.block_count"].contents() == layers
    assert len(check.tensors) == len(keep) + len(extra), (len(check.tensors), len(keep))
    print(
        f"wrote {dst}: {len(check.tensors)} tensors, block_count {layers}, {dst.stat().st_size / 1e9:.2f} GB"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("layers", type=int)
    ap.add_argument("--head", type=Path)
    a = ap.parse_args()
    main(a.src, a.dst, a.layers, a.head)
