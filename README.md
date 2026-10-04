# Recovery ML Container

This container trains and evaluates ML models on the empirical recovery dataset.

## Included models

1. Logistic Regression
2. Random Forest
3. XGBoost

The pipeline uses GroupKFold by `incident_group_id` to reduce leakage between the five candidate actions belonging to the same incident group.

## Features

Only information available before recovery action is used:

- fault_type
- severity
- workload_profile
- candidate_action
- baseline_* telemetry
- pre_* telemetry
- recovery thresholds
- action_elapsed_ms

`post_*` variables are deliberately excluded because they are outcomes observed after the recovery action.

## Target

`recovery_success`

## Run on Windows / Docker Desktop

Open PowerShell in this folder:

```powershell
docker compose build
docker compose run --rm recovery-ml
```

Or:

```powershell
docker compose up --build
```

Results will appear in:

```text
results\
```

## Expected output

- model_comparison.csv
- oof_policy_results.csv
- oof_policy_summary.csv
- action_probability_scores.csv
- feature_importance.csv
- shap_global_importance.csv
- roc_curves.png
- confusion_matrix_best_model.png
- feature_importance.png
- shap_summary.png
- trained_best_model.joblib
- model_metadata.json

## Important

The current dataset contains the available 968 empirical records. Do not add synthetic rows to reach 1050.

The policy evaluation is out-of-fold: each incident group is scored by a model that did not train on that incident group.

## Optional custom dataset

Replace:

```text
data/recovery_experiments_raw.csv
```

with your updated CSV, preserving the column names.
