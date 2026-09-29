"""Replay representation validation and retain only aggregate loss components."""
import argparse
import gc
from pathlib import Path
import statistics
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "src"))
    import torch
    from responsewm import training
    from responsewm.data import ManifestStore
    from responsewm.io import seed_all, write_json

    store = ManifestStore(args.manifest)
    original = training.representation_loss
    rows = []

    def capture(model, inp, sup):
        loss, terms = original(model, inp, sup)
        rows.append(terms)
        return loss, terms

    training.representation_loss = capture
    report = {}
    for name in ("best", "last"):
        rows.clear()
        model, payload = training.load_trained(args.run / "representation" / f"{name}.pt", "cuda")
        store.set_statistics(payload["metadata"]["statistics"])
        seed_all(model.cfg.training.seed, model.cfg.training.threads, model.cfg.training.strict_determinism)
        metrics = training.validate(model, store, model.cfg, "representation")
        report[name] = {
            "step": payload["step"],
            "validation": metrics,
            "component_means": {key: statistics.mean(row[key] for row in rows) for key in rows[0]},
        }
        write_json(args.output, report)
        print(name, report[name], flush=True)
        del model, payload
        gc.collect()
        torch.cuda.empty_cache()
    training.representation_loss = original


if __name__ == "__main__":
    main()
