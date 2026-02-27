"""Export 5 ensemble OTSTransformer checkpoints to ONNX for browser inference.

Wraps the model to produce two outputs: logits (1,90) and embedding (1,128).
The embedding output is the pooled representation BEFORE the classification head,
used for retrieval queries client-side.

Usage:
    uv run python scripts/export_onnx.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from turnzero.models.transformer import ModelConfig, OTSTransformer


class OTSExportWrapper(nn.Module):
    """Wrapper that returns (logits, embedding) and avoids dynamic tensor creation."""

    def __init__(self, model: OTSTransformer) -> None:
        super().__init__()
        self.model = model
        # Pre-register side indices as buffers (ONNX needs static tensors)
        self.register_buffer("side_a", torch.tensor(0, dtype=torch.long))
        self.register_buffer("side_b", torch.tensor(1, dtype=torch.long))

    def forward(self, team_a: torch.Tensor, team_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.model
        # Cast int32 → int64 for embedding lookups (ONNX Runtime Web WASM needs int32 inputs)
        team_a = team_a.long()
        team_b = team_b.long()
        # Embed teams using pre-registered side buffers
        e_a = (
            m.emb_species(team_a[:, :, 0])
            + m.emb_item(team_a[:, :, 1])
            + m.emb_ability(team_a[:, :, 2])
            + m.emb_tera(team_a[:, :, 3])
            + m.emb_move(team_a[:, :, 4])
            + m.emb_move(team_a[:, :, 5])
            + m.emb_move(team_a[:, :, 6])
            + m.emb_move(team_a[:, :, 7])
            + m.emb_side(self.side_a)
        )
        e_b = (
            m.emb_species(team_b[:, :, 0])
            + m.emb_item(team_b[:, :, 1])
            + m.emb_ability(team_b[:, :, 2])
            + m.emb_tera(team_b[:, :, 3])
            + m.emb_move(team_b[:, :, 4])
            + m.emb_move(team_b[:, :, 5])
            + m.emb_move(team_b[:, :, 6])
            + m.emb_move(team_b[:, :, 7])
            + m.emb_side(self.side_b)
        )

        tokens = torch.cat([e_a, e_b], dim=1)  # (B, 12, d_model)
        tokens = m.encoder(tokens)
        pooled = tokens.mean(dim=1)  # (B, d_model) — always mean pool

        logits = m.head(pooled)  # (B, 90)
        return logits, pooled


def export_one(ckpt_path: Path, out_path: Path) -> None:
    device = torch.device("cpu")
    model = OTSTransformer.load_from_checkpoint(ckpt_path, device)
    wrapper = OTSExportWrapper(model)
    wrapper.eval()

    # Dummy inputs — int32 for ONNX Runtime Web WASM compatibility
    team_a = torch.zeros(1, 6, 8, dtype=torch.int32)
    team_b = torch.zeros(1, 6, 8, dtype=torch.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (team_a, team_b),
        str(out_path),
        opset_version=17,
        input_names=["team_a", "team_b"],
        output_names=["logits", "embedding"],
        dynamic_axes={
            "team_a": {0: "batch"},
            "team_b": {0: "batch"},
            "logits": {0: "batch"},
            "embedding": {0: "batch"},
        },
    )

    # Validate ONNX model
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    # Validate against PyTorch outputs
    with torch.no_grad():
        pt_logits, pt_emb = wrapper(team_a, team_b)

    sess = ort.InferenceSession(str(out_path))
    ort_out = sess.run(None, {
        "team_a": team_a.numpy().astype("int32"),
        "team_b": team_b.numpy().astype("int32"),
    })

    np.testing.assert_allclose(ort_out[0], pt_logits.numpy(), atol=1e-5)
    np.testing.assert_allclose(ort_out[1], pt_emb.numpy(), atol=1e-5)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  {out_path.name}: {size_mb:.1f} MB — validated OK")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ensemble_dir = root / "outputs" / "runs"
    out_dir = root / "web" / "public" / "models"

    ckpt_dirs = sorted(ensemble_dir.glob("ensemble_*"))
    if not ckpt_dirs:
        raise FileNotFoundError(f"No ensemble_* dirs in {ensemble_dir}")

    print(f"Exporting {len(ckpt_dirs)} ensemble members to ONNX...")
    for i, d in enumerate(ckpt_dirs, 1):
        ckpt_path = d / "best.pt"
        out_path = out_dir / f"ensemble_{i:03d}.onnx"
        print(f"\n[{i}/{len(ckpt_dirs)}] {d.name}")
        export_one(ckpt_path, out_path)

    total_mb = sum(f.stat().st_size for f in out_dir.glob("*.onnx")) / 1024 / 1024
    print(f"\nDone! Total ONNX size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
