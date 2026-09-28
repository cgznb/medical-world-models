# Gastric aggregate results

These files report internal research experiments completed on 2026-09-28.
They contain aggregate metrics and figures, not patient data or trained models.
Source and reproducible execution paths are in
[gastric_tcwm_upgrade](../../gastric_tcwm_upgrade).

| Study | Main result | Interpretation |
| --- | --- | --- |
| [Repair](repair/REPAIR_REPORT_ZH.md) | Locked internal test recurrence AUROC 0.6240; clinical-only 0.6253 | Improvement over the initial adaptation is mainly clinical; CT increment not established. |
| [CT development](ct_optimization/CT_REPORT_ZH.md) | Nested OOF S1 AUROC 0.5989; clinical-only 0.5892; paired difference interval crosses zero | No reliable CT improvement. Final fixed-budget refit validation AUROC 0.5196; candidate not promoted. |

The repair study scored the original 65-patient test partition once after
locking. The later CT study used only the original 521 training patients for
candidate comparison and did not score the original test again. OOF scores
were used for development/selection and are not an independent confirmation.
The cohort is not an external clinical validation cohort.

Published artifacts are protocols, aggregate summaries, descriptive final
validation metrics, paired-bootstrap uncertainty, CT dependency diagnostics,
and charts. Patient identifiers, split lists, raw images, clinical tables,
feature caches, individual predictions, weights, private contracts/locks,
raw JUnit XML, and unreviewed execution manifests are excluded. Bundle hashes
in the historical aggregate files are artifact fingerprints, not patient IDs.

Public-copy engineering verification is summarized in
[test_summary.json](test_summary.json). Synthetic tests establish execution
properties and information boundaries, not medical effectiveness.
