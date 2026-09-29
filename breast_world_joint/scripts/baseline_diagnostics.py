"""Evaluate existing clinical and observed-only branches, without fitting models."""
import argparse
from pathlib import Path
import sys
import gc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "src"))
    import numpy as np
    import torch
    from responsewm import training
    from responsewm.data import ManifestStore
    from responsewm.io import seed_all, autocast, write_json
    from responsewm.metrics import classification_metrics

    store = ManifestStore(args.manifest)
    report = {"protocol": "Existing fitted clinical prior and observed-only branch of trained checkpoints; no new training", "branches": {}}
    for stage in ("readout", "joint"):
        model, payload = training.load_trained(args.run / stage / "best.pt", "cuda")
        store.set_statistics(payload["metadata"]["statistics"])
        seed_all(model.cfg.training.seed, model.cfg.training.threads, model.cfg.training.strict_determinism)
        y, prior, observed = [], [], []
        with torch.no_grad():
            for index in training._validation_indices(store, model.cfg.training.validation_cases):
                inp = store.batch([index], "cuda", supervised=False)
                # The clinical prior uses FP32 arithmetic; the MRI branch matches the run's BF16 forward.
                prior.append(float(model.pcr.prior(inp.clinical, inp.clinical_mask).float().sigmoid()[0]))
                with autocast("cuda", model.cfg.training.precision):
                    states = model.encode_sequence(inp.observed, inp.observed_days, inp.observed_mask)
                    memory = model.memory(inp, states)
                    logit = model.pcr.observed_logit(memory, inp.clinical, inp.clinical_mask)
                observed.append(float(logit.float().sigmoid()[0]))
                y.append(store.cases[index]["target"]["pcr"])
        y = np.asarray(y)
        for label, values in (("clinical_prior", prior), ("observed_only", observed)):
            p = np.asarray(values)
            pred = p >= 0.5
            metrics = classification_metrics(y, p)
            metrics.update(accuracy=float((pred == y).mean()), sensitivity=float(pred[y == 1].mean()), specificity=float((~pred[y == 0]).mean()))
            report["branches"][stage + "/" + label] = metrics
        write_json(args.output, report)
        print(stage, {key: {metric: report["branches"][stage + "/" + key][metric] for metric in ("auroc", "auprc", "nll", "accuracy")} for key in ("clinical_prior", "observed_only")}, flush=True)
        del model, payload
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
