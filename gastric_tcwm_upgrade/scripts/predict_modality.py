#!/usr/bin/env python
"""Predict from a private, native modality-event-v2 tensor batch."""
from pathlib import Path
import argparse
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from stageworld_tcwm.modality_inference import ModalityPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    os.umask(0o077)
    payload = torch.load(args.batch, map_location="cpu", weights_only=True)
    if payload.get("schema") != "modality-event-v2":
        raise ValueError("Input must explicitly declare modality-event-v2")
    predictor = ModalityPredictor.load(args.checkpoint, args.device)
    result = predictor.predict(payload["tensors"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as output:
        torch.save(result, output)
    print(f"Saved {result['risk'].shape[0]} patients, {result['risk'].shape[1]} query slots")


if __name__ == "__main__":
    main()
