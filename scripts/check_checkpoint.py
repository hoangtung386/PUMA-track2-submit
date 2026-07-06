"""Fail-fast compatibility check used before building the Docker image."""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path

from prometheus.api import build_model, load_config
from prometheus.checkpoint import assert_checkpoint_compatible, load_checkpoint, select_inference_state


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/submission.toml"))
    parser.add_argument("--nuclei", type=Path, default=Path("models/best_primary.ckpt"))
    parser.add_argument("--tissue", type=Path, default=Path("models/best_tissue.ckpt"))
    args = parser.parse_args()

    config = load_config(args.config)
    for task, path in (("nuclei", args.nuclei), ("tissue", args.tissue)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {task} checkpoint: {path}")
        checkpoint = load_checkpoint(path, "cpu")
        assert_checkpoint_compatible(checkpoint, config)
        model = build_model(config)
        model.load_state_dict(select_inference_state(checkpoint), strict=True)
        state_name = "EMA" if checkpoint.get("ema_state") is not None else "raw"
        metrics = checkpoint.get("metrics", {})
        print(f"task={task}")
        print(f"checkpoint={path}")
        print(f"sha256={sha256(path)}")
        print(f"epoch={checkpoint.get('epoch')}")
        print(f"inference_state={state_name}")
        print(f"nuclei_f1={metrics.get('nuclei/macro_f1_summed')}")
        print(f"tissue_dice={metrics.get('tissue/dice_mean_fg')}")
        del model, checkpoint
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
