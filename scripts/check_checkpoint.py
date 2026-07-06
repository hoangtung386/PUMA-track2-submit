"""Fail-fast compatibility check used before building the Docker image."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--checkpoint", type=Path, default=Path("models/prometheus.ckpt"))
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    config = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    assert_checkpoint_compatible(checkpoint, config)
    model = build_model(config)
    model.load_state_dict(select_inference_state(checkpoint), strict=True)
    state_name = "EMA" if checkpoint.get("ema_state") is not None else "raw"
    print(f"checkpoint={args.checkpoint}")
    print(f"sha256={sha256(args.checkpoint)}")
    print(f"inference_state={state_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
