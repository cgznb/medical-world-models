"""Export the dated gastric development study through explicit field allowlists.

Private source reports must stay outside the public repository. Run with
``--source STUDY_DIRECTORY --output results/gastric/next_round_20260929``.
No models, patient records, contract objects, or small calibration bins are copied.
"""

import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


ARMS = ("G0", "G1", "G2", "G3")
STAGES = ("S0", "S1", "S2")
METRICS = ("nll", "brier", "auroc", "average_precision")
METRIC_KEYS = ("n", "positive", *METRICS)
TRAINING_KEYS = (
    "selected_epoch", "best_validation_nll", "parameter_count", "test_evaluated",
    "epochs_completed", "optimizer_steps", "supervised_steps",
    "selected_optimizer_steps", "selected_supervised_steps", "steps_per_full_epoch",
    "optimizer_step_budget", "selection_policy", "selected_kind", "baseline0_won",
    "stage_weights", "samples_eval", "mc_seed_policy", "mc_antithetic",
    "training_patients_seen", "train_patients", "train_events", "planned_epochs",
    "planned_optimizer_steps", "actual_epochs", "actual_optimizer_steps",
    "budget_fulfilled", "supervised_step_budget", "validation_evaluations",
    "final_validation_nll", "validation_used_for_checkpoint_selection",
    "early_stopping_unit", "stop_reason",
)
FIT_KEYS = (
    "objective", "dtype", "solver", "lambda", "max_iter", "iterations",
    "function_evaluations", "objective_value", "gradient_l2_norm", "gradient_max_abs",
    "converged", "convergence_definition", "training_patients", "training_events",
    "features", "selection_nll", "training_prevalence",
)
PROVENANCE_KEYS = (
    "rank", "fit_patients", "projection_observations",
    "projection_fit", "feature_level", "normalization", "missing_images",
    "tabular_statistics_sha256", "different_ranks_define_different_target_spaces",
    "target_normalization", "target_moment_statistics_sha256",
    "paired_training_patients", "ridge_lambda",
    "ridge_objective", "ridge_selection", "gaussian_covariance",
)


def pick(value, keys):
    return {key: value[key] for key in keys if key in value}


def metric(value):
    result = pick(value, METRIC_KEYS)
    if "calibration" in value:
        result["calibration"] = pick(value["calibration"], (
            "mean_prediction", "observed_frequency", "mean_prediction_minus_observed",
            "expected_calibration_error",
        ))
    return result


def stage_report(value):
    result = pick(value, (
        "interpretation", "causal_effects_identified", "selection_nll",
        "legacy_three_stage_nll", "stage_weights", "selection_definition",
        "evaluated", "used_for_checkpoint_selection",
    ))
    result["stages"] = {stage: metric(value["stages"][stage]) for stage in STAGES}
    return result


def training(value):
    result = pick(value, TRAINING_KEYS)
    result["optimizer_groups"] = [
        pick(group, ("name", "learning_rate", "parameter_count"))
        for group in value["optimizer_groups"]
    ]
    result["task_counts"] = {
        name: pick(value["task_counts"][name], ("effective_updates", "effective_targets"))
        for name in ("endpoint", "pcr", "ct", "kl", "flow", "prior", "prior_ct",
                     "observation_recon", "readout_l2")
    }
    result["baseline_candidates"] = []
    for candidate in value["baseline_candidates"]:
        exported = pick(candidate, ("name", "selected", "optimizer_steps", "supervised_steps"))
        exported["validation"] = stage_report(candidate["validation"])
        result["baseline_candidates"].append(exported)
    return result


def paired(value):
    return {
        name: pick(value[name], ("candidate_minus_reference", "fixed_oof_paired_bootstrap_95ci"))
        for name in METRICS
    }


def source_hashes(value):
    for name, digest in value.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid source provenance entry")
    return value


def provenance(value):
    result = pick(value, PROVENANCE_KEYS)
    for key in ("target_space_id", "projection_space_id"):
        if key in value:
            result[key] = value[key]
    result["encoder_provenance"] = pick(value.get("encoder_provenance", {}), (
        "clinical_transform_sha256", "treatment_support_sha256",
        "fit_ids_exactly_match_inner_train",
    ))
    return result


def export_summary(value, statuses):
    result = pick(value, (
        "development_only", "independent_confirmation", "original_validation_or_test_scored",
        "samples", "mc_seed", "mc_seed_policy", "mc_antithetic", "stage_weights",
        "outcome", "uncertainty_limit",
    ))
    result.update({
        "schema": "public-gastric-next-round-summary-v1",
        "study_date": "2026-09-29",
        "evaluation_scope": "reused_original_training_three_fold_OOF_development",
        "training_runs": len(statuses),
        "all_runs_completed_successfully": all(row["exit_code"] == 0 for row in statuses),
        "total_actual_supervised_steps": sum(row["training"]["supervised_steps"] for row in statuses),
        "clinical_anchor": metric(value["clinical_anchor"]),
        "clinical_anchor_regularization_C": 1,
        "clinical_anchor_distinct_from_selected_P1_B1": True,
        "reliable_incremental_benefit_established": False,
        "old_research_checkpoint_replaced": False,
        "additional_training_seeds_triggered": False,
        "locked_protocol": {
            "training_seed": 17,
            "training_mc_samples": 4,
            "selection_mc_seed": 10017,
            "selection_mc_samples": 64,
            "validation_interval_supervised_steps": 50,
            "early_stopping_patience_validation_cycles": 5,
            "maximum_supervised_steps_per_run": 1000,
            "paired_bootstrap_draws": 2000,
            "paired_bootstrap_seed": 20260929,
            "paired_bootstrap_sampling": "patient_paired_outcome_stratified_fixed_OOF",
        },
        "historical_score_comparability": "Stage weights, MC and training schedule changed; historical differences cannot identify one modification's effect.",
        "cases": {},
    })
    for arm in ARMS:
        case = value["cases"][arm]
        exported = {
            "oof": {stage: metric(case["oof"][stage]) for stage in STAGES},
            "S1_vs_clinical_anchor": paired(case["S1_vs_clinical_anchor"]),
            "folds": [],
        }
        exported["oof"].update(pick(case["oof"], ("selection_nll", "legacy_three_stage_nll")))
        for fold in case["folds"]:
            row = pick(fold, (
                "fold", "train_patients", "train_events", "outer_patients", "outer_events",
                "rank", "target_space", "stage_weights", "seed", "mc_samples",
                "changed_evaluation_contract_from_historical",
            ))
            row["information_sets"] = pick(fold["information_sets"], STAGES)
            row["training"] = training(fold["training"])
            row["outer"] = stage_report(fold["outer"])
            exported["folds"].append(row)
        result["cases"][arm] = exported
    for comparison in ("G1_vs_G0", "G2_vs_G1", "G3_vs_G1"):
        result[comparison] = {stage: paired(value[comparison][stage]) for stage in ("S0", "S1")}
    return result


def diagnostic_comparison(value):
    result = pick(value, ("bootstrap_draws", "bootstrap_seed", "uncertainty_scope"))
    result["delta_new_minus_reference"] = pick(value["delta_new_minus_reference"], METRICS)
    result["folds_improved"] = pick(value["folds_improved"], METRICS)
    result["bootstrap_95_percent"] = {
        name: pick(value["bootstrap_95_percent"][name], ("low", "high", "valid_draws"))
        for name in METRICS
    }
    result["folds"] = [
        {**pick(fold, ("fold", "patients")),
         "delta_new_minus_reference": pick(fold["delta_new_minus_reference"], METRICS)}
        for fold in value["folds"]
    ]
    return result


def export_diagnostics(value):
    result = pick(value, (
        "study_scope", "original_validation_or_test_scored", "full_outer_oof", "fold_indices",
        "ranks", "lambda_grid", "selection", "endpoint", "scenario_semantics", "B4_scope",
        "method_changed_from_historical_direct_ct", "limitations", "elapsed_seconds",
    ))
    result["schema"] = "public-gastric-fixed-feature-diagnostic-v1"
    result["source_sha256"] = source_hashes(value["source_sha256"])
    result["calibration_bins_omitted"] = "Small bins can disclose individual predictions and labels."
    result["results"] = {}
    candidates = []
    for rank in ("8", "32"):
        block = value["results"][rank]
        exported = {
            "next_step": block["next_step"],
            "pooled_outer": {name: metric(block["pooled_outer"][name]) for name in ("B0", "B1", "B2", "B3", "B4")},
            "paired_comparisons": {
                name: diagnostic_comparison(block["paired_comparisons"][name])
                for name in ("B2_vs_B1", "B3_vs_B2", "B4_vs_B3")
            },
            "folds": [],
        }
        for fold in block["folds"]:
            row = pick(fold, (
                "fold", "rank", "training_patients", "training_events", "selection_patients",
                "outer_patients", "outer_events", "stage_weights", "stage_weights_reason",
                "seed", "mc_samples", "baseline0_selected",
            ))
            row["provenance"] = provenance(fold["provenance"])
            row["models"] = {}
            for name in ("B0", "B1", "B2", "B3", "B4"):
                model = fold["models"][name]
                row["models"][name] = {
                    "information_set": model["information_set"],
                    "selected_fit": pick(model["selected_fit"], FIT_KEYS),
                    "lambda_candidates": [pick(fit, FIT_KEYS) for fit in model["lambda_candidates"]],
                    "selection": metric(model["selection"]),
                    "outer": metric(model["outer"]),
                }
                if name != "B0":
                    candidates.extend(model["lambda_candidates"])
            for key in ("supervised_optimizer_updates", "selected_steps"):
                row[key] = pick(fold[key], ("B0", "B1", "B2", "B3", "B4"))
            exported["folds"].append(row)
        result["results"][rank] = exported
    result["regularized_candidate_count"] = len(candidates)
    result["all_regularized_candidates_converged"] = all(row["converged"] for row in candidates)
    return result


def forecast_metrics(value):
    return {
        "points": {
            name: pick(value["points"][name], (
                "patients", "elements", "sse", "training_mean_sse", "mse",
                "skill_vs_training_CT1_mean",
            ))
            for name in ("training_CT1_mean", "copy_CT0", "ridge_CT0_condition",
                         "prior_single_sample", "prior_decoded_MC_mean")
        },
        "distributions": {
            name: pick(value["distributions"][name], (
                "energy_score", "marginal_90pct_coverage", "marginal_90pct_width",
                "mean_projected_variance",
            ))
            for name in ("prior", "training_Gaussian")
        },
        "sets_and_channel_moments": pick(value["sets_and_channel_moments"], (
            "prior_single_sample_set_loss", "prior_expected_sample_set_loss", "copy_CT0_set_loss",
            "training_channel_mean_degenerate_set_loss", "prior_MC_channel_mean_mse",
            "prior_MC_channel_sd_mse", "prior_single_channel_mean_mse", "prior_single_channel_sd_mse",
        )),
    }


def export_forecast(value):
    result = pick(value, (
        "endpoint", "development_evidence_only", "original_validation_or_test_scored", "ranks",
        "samples", "mc_seed", "mc_seed_policy", "batch_size", "device", "fold_indices",
        "full_outer_oof", "forward_inputs", "point_forecast", "set_scoring", "energy_score",
        "intervals", "limitations", "elapsed_seconds",
    ))
    result["schema"] = "public-gastric-prior-forecast-evaluation-v1"
    result["checkpoint_scope"] = "historical_selected_token_prior_folds_before_G0_G1_G2_G3"
    result["source_sha256"] = source_hashes(value["source_sha256"])
    result["results"] = {}
    for rank in ("8", "32"):
        block = value["results"][rank]
        exported = {"pooled_outer": {"patients": block["pooled_outer"]["patients"],
                                      **forecast_metrics(block["pooled_outer"])}, "folds": []}
        for fold in block["folds"]:
            row = pick(fold, ("fold", "paired_outer_patients", "training_patients", "selected_epoch_zero_based"))
            row["provenance"] = provenance(fold["provenance"])
            row.update(forecast_metrics(fold))
            exported["folds"].append(row)
        result["results"][rank] = exported
    return result


def export_dependency(value, verification):
    result = pick(value, (
        "model_checkpoints", "all_frozen_anchors_and_preprocessing_unchanged",
        "all_exports_exactly_match_selected_best", "total_actual_supervised_steps", "partition",
        "patients_per_checkpoint", "samples", "mc_seed", "permutation_seed", "mc_seed_policy",
        "mc_antithetic", "stage_weights", "original_validation_or_test_scored",
        "outer_evaluation_scored", "causal_effects_identified", "independent_confirmation",
        "all_prefix_checks_passed", "limitations",
    ))
    result["schema"] = "public-gastric-next-round-dependency-v1"
    boundaries = ("S0_future_target_change", "pCR_S0_future_target_change",
                  "S1_S2_without_new_information", "undeclared_future_treatment_change")
    result["max_boundary_difference"] = pick(value["max_boundary_difference"], boundaries)
    result["arms"] = {
        arm: pick(value["arms"][arm], (
            "CT0_S0_probability_mean_absolute_difference", "CT1_S1_probability_mean_absolute_difference",
            "CT1_permutation_minus_original_weighted_nll", "CT1_permutation_worsened_nll_folds",
            "CT1_permutation_improved_nll_folds",
        )) for arm in ARMS
    }
    result["records"] = []
    for record in value["records"]:
        row = pick(record, ("case", "fold", "role", "patients", "original_test_or_validation_scored", "stage_weights"))
        row["health"] = pick(record["health"], ("raw_kl", "free_nats_kl", "prior_active_units", "posterior_active_units"))
        dep = record["dependency"]
        row["dependency"] = pick(dep, ("samples", "seed", "permutation_seed", "mc_seed_policy", "mc_antithetic"))
        row["dependency"]["prefix_boundary_checks"] = {
            **pick(dep["prefix_boundary_checks"], ("no_new_information_patients", "passed")),
            "max_absolute_logit_or_rate_difference": pick(dep["prefix_boundary_checks"]["max_absolute_logit_or_rate_difference"], boundaries),
        }
        row["dependency"]["perturbations"] = {
            name: {**pick(dep["perturbations"][name], ("eligible_patients", "selection_nll_difference")),
                   "stages": {stage: pick(dep["perturbations"][name]["stages"][stage], ("mean_absolute_difference", "max_absolute_difference")) for stage in STAGES}}
            for name in ("CT0", "CT1", "paired_CT", "known_scenario")
        }
        result["records"].append(row)
    result["checkpoint_verification"] = {
        "total_actual_supervised_steps": verification["total_actual_supervised_steps"],
        "records": [pick(row, (
            "case", "fold", "frozen_anchor_and_preprocessing_unchanged", "export_exactly_selected_best",
            "actual_supervised_steps", "selected_supervised_steps", "selected_kind",
        )) for row in verification["records"]],
    }
    return result


def export_mc(value):
    result = pick(value, ("role", "patients", "original_test_or_validation_scored",
                          "independent_clinical_validation", "stage_weights"))
    result["schema"] = "public-gastric-mc-stability-v1"
    result["checkpoint_scope"] = "historical_selected_token_prior_fold_0"
    result["mc"] = pick(value["mc"], (
        "patients", "mc_seed_policy", "mc_antithetic", "stage_weights", "min_delta",
        "recommended_samples_eval", "recommendation_rule", "seed_selection_permitted",
        "independent_clinical_validation",
    ))
    result["mc"]["records"] = [pick(row, (
        "samples", "seeds", "selection_nll", "mc_nll_sd", "sd_over_min_delta",
    )) for row in value["mc"]["records"]]
    return result


def export_engineering(value):
    result = pick(value, (
        "full_test_count", "full_test_status", "inner_validation_patients_checked",
        "selected_kind", "completed_resume_recovery_best_report_byte_identical", "optimizer_steps",
        "supervised_steps", "torch_version", "gpu", "source_manifest_scope",
        "outer_fold_or_original_holdout_scored",
    ))
    result["schema"] = "public-gastric-engineering-checks-v1"
    result["evidence_scope"] = "original_training_environment_before_publication"
    for key in ("inference_max_absolute_difference", "neural_last_inference_max_absolute_difference"):
        result[key] = pick(value[key], STAGES)
    result["observation_reconstruction_gradients"] = [
        {name: pick(row[name], ("present", "norm")) for name in (
            "prior", "observation_attention", "observation_gate", "decoder", "outcome",
        )} for row in value["observation_reconstruction_gradients"]
    ]
    result["source_sha256"] = source_hashes(value["source_sha256"])
    result["source_hash_scope"] = "original_training_source; public path adaptations may change file hashes"
    return result


def validate_public(value):
    blocked_keys = {"contract_id", "bundle_sha256", "patient_id", "patient_ids", "case_id",
                    "case_ids", "train_ids", "test_ids", "validation_ids", "fixed_probability_bins",
                    "fit_patients_hash", "paired_fit_patients_hash", "original_training_membership_sha256"}
    if isinstance(value, dict):
        if blocked_keys.intersection(value):
            raise ValueError("Unexpected private field in aggregate export")
        for item in value.values():
            validate_public(item)
    elif isinstance(value, list):
        for item in value:
            validate_public(item)
    elif isinstance(value, str) and re.search(r"/(?:home|data1|root)/", value):
        raise ValueError("Unexpected deployment path in aggregate export")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publication-test-report", type=Path)
    args = parser.parse_args()

    def load(relative):
        return json.loads((args.source / relative).read_text(encoding="utf-8"))

    statuses = load("runs/study_status.json")
    outputs = {
        "summary.json": export_summary(load("evaluation/report.json"), statuses),
        "diagnostic_baselines.json": export_diagnostics(load("diagnostics/report.json")),
        "forecast_prior.json": export_forecast(load("forecast_prior/report.json")),
        "dependency.json": export_dependency(load("dependency/summary.json"), load("dependency/checkpoint_verification.json")),
        "mc_stability.json": export_mc(load("mc_probe/report.json")),
        "engineering_checks.json": export_engineering(load("smoke_checks.json")),
        "training_status.json": {
            "schema": "public-gastric-next-round-training-status-v1",
            "records": [{**pick(row, ("case", "fold", "exit_code")), "training": training(row["training"])} for row in statuses],
        },
    }
    if args.publication_test_report:
        root = ET.parse(args.publication_test_report).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        checks = {name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
                  for name in ("tests", "failures", "errors", "skipped")}
        checks["elapsed_seconds"] = sum(float(suite.attrib.get("time", 0)) for suite in suites)
        checks["passed"] = checks["tests"] - checks["failures"] - checks["errors"] - checks["skipped"]
        checks["scope"] = "public_gastric_copy_full_pytest_suite"
        outputs["engineering_checks.json"]["public_copy_test"] = checks
    for value in outputs.values():
        validate_public(value)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported {len(outputs)} aggregate reports.")


if __name__ == "__main__":
    main()
