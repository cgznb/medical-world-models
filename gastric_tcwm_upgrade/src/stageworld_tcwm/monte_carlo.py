"""Evaluation-only random numbers; case keys never enter model features."""
import hashlib
import json

import torch


def anonymous_case_key(identifier):
    return hashlib.sha256(("tcwm-case-v1:" + str(identifier)).encode()).hexdigest()


def cohort_case_keys(cohort, rows):
    return [anonymous_case_key(cohort.ids[int(row)]) for row in rows]


def case_key_epsilon(case_keys, samples, dimension, seed=17, *, antithetic=False,
                     device="cpu", dtype=torch.float32):
    """Draw on CPU for a stable case/draw contract across batching and devices.

    Antithetic draws are consecutive pairs, so extending K preserves earlier draws.
    Hashes are internal random-number keys and are never prediction inputs.
    """
    if samples < 1 or dimension < 1 or not case_keys:
        raise ValueError("Require case keys and positive sample/latent dimensions")
    if any(not isinstance(key, str) or not key for key in case_keys):
        raise ValueError("Case keys must be nonempty strings")
    if len(set(case_keys)) != len(case_keys):
        raise ValueError("Case keys must be unique within an evaluation batch")
    if antithetic and samples % 2:
        raise ValueError("Antithetic sampling requires an even number of draws")
    output = torch.empty((len(case_keys), samples, dimension), dtype=torch.float32)
    for patient, key in enumerate(case_keys):
        for draw in range(samples // 2 if antithetic else samples):
            payload = json.dumps(["tcwm-mc-v1", int(seed), key, draw], separators=(",", ":"))
            draw_seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63)
            epsilon = torch.randn(dimension, generator=torch.Generator().manual_seed(draw_seed))
            if antithetic:
                output[patient, 2 * draw] = epsilon
                output[patient, 2 * draw + 1] = -epsilon
            else:
                output[patient, draw] = epsilon
    return output.to(device=device, dtype=dtype)


def evaluation_epsilon(model, batch_size, samples, seed, *, policy="batch_start",
                       case_keys=None, antithetic=False):
    if policy not in ("batch_start", "case_key"):
        raise ValueError("Unknown Monte Carlo seed policy")
    if samples < 1 or (antithetic and samples % 2):
        raise ValueError("Require positive draws; antithetic draws must be even")
    device = next(model.parameters()).device
    dimension = getattr(model, "state_dim", model.cfg.latent_dim)
    if policy == "case_key":
        if case_keys is None or len(case_keys) != batch_size:
            raise ValueError("case_key evaluation requires one anonymous case key per row")
        return case_key_epsilon(case_keys, samples, dimension, seed, antithetic=antithetic, device=device)
    if not antithetic:
        return None  # Preserve historical device-local seeded sampling exactly.
    draws = torch.randn((batch_size, samples // 2, dimension), device=device,
                        generator=torch.Generator(device=device).manual_seed(seed))
    return torch.stack((draws, -draws), dim=2).flatten(1, 2)


def monte_carlo_standard_error(values, antithetic=False):
    if not antithetic:
        return values.std(1, unbiased=False) / values.shape[1]**.5
    pairs = values.reshape(values.shape[0], values.shape[1] // 2, 2, *values.shape[2:]).mean(2)
    if pairs.shape[1] < 2:
        return None
    return pairs.std(1, unbiased=True) / pairs.shape[1]**.5
