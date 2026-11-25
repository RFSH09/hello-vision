"""Fine-tune a DINOv2-small embedding with an ArcFace-style metric-learning head.

NOT RUN YET: written for the first real training session. Needs the optional extra
``uv sync --extra train`` (torch, torchvision) and the DINOv2 weights downloaded after their
license has been recorded in docs/LICENSES.md. Data: per-SKU crop folders produced by
``training.build_embedding_dataset`` (real crops + packshot augmentations).

Usage: ``uv run python -m training.train_embedding --config training/configs/embedding.yaml``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml

from training.common import CsvLogger, set_seed


def main(argv: Sequence[str] | None = None) -> int:
    """Train and export the embedding model to ONNX."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    import torch
    import torch.nn.functional as functional
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    tf = transforms.Compose(
        [
            transforms.Resize((cfg["input_size"], cfg["input_size"])),
            transforms.ColorJitter(0.4, 0.4, 0.3, 0.05),
            transforms.RandomAffine(8, scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    ds = datasets.ImageFolder(cfg["crops_dir"], transform=tf)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)

    backbone = torch.hub.load("facebookresearch/dinov2", cfg["backbone"])  # downloads weights
    dim = int(backbone.embed_dim)
    centers = nn.Parameter(torch.randn(len(ds.classes), dim) * 0.01)
    opt = torch.optim.AdamW([*backbone.parameters(), centers], lr=cfg["lr"])
    logger = CsvLogger(Path(cfg["out_dir"]) / "metrics.csv")
    scale, margin = float(cfg["arcface_scale"]), float(cfg["arcface_margin"])

    for epoch in range(int(cfg["epochs"])):
        total = 0.0
        for x, y in loader:
            emb = functional.normalize(backbone(x))
            cos = emb @ functional.normalize(centers).T
            theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
            target = torch.cos(theta + margin)
            logits = scale * torch.where(functional.one_hot(y, cos.shape[1]).bool(), target, cos)
            loss = functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(y)
        logger.log(epoch=epoch, loss=total / len(ds))

    backbone.eval()
    dummy = torch.zeros(1, 3, cfg["input_size"], cfg["input_size"])
    torch.onnx.export(backbone, dummy, str(Path(cfg["out_dir"]) / "embedding.onnx"),
                      input_names=["input"], output_names=["embedding"])  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
