from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path("/app")
DATA = ROOT / "data" / "recovery_experiments_raw.csv"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "recovery_success"
GROUP = "incident_group_id"

CATEGORICAL = [
    "fault_type",
    "severity",
    "workload_profile",
    "candidate_action",
]

NUMERIC_CANDIDATES = [
    "baseline_cpu_pct",
    "baseline_memory_mb",
    "baseline_latency_ms",
    "baseline_error_rate_pct",
    "baseline_request_rate",
    "baseline_availability",
    "baseline_inference_latency_ms",
    "baseline_queue_length",
    "pre_cpu_pct",
    "pre_memory_mb",
    "pre_latency_ms",
    "pre_error_rate_pct",
    "pre_request_rate",
    "pre_availability",
    "pre_inference_latency_ms",
    "pre_queue_length",
    "latency_limit_ms",
    "error_limit_pct",
    "availability_min",
    "action_elapsed_ms",
]


def make_preprocessor(cat_cols, num_cols):
    return ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
        ],
        verbose_feature_names_out=True,
    )


def evaluate_models(X, y, groups, models, cv):
    rows = []
    roc = {}
    oof_probabilities = {}

    for name, estimator in models.items():
        pipe = Pipeline(
            [
                ("preprocess", make_preprocessor(CATEGORICAL, NUMERIC)),
                ("model", estimator),
            ]
        )

        probabilities = cross_val_predict(
            pipe,
            X,
            y,
            cv=cv,
            groups=groups,
            method="predict_proba",
        )[:, 1]

        predictions = (probabilities >= 0.5).astype(int)

        rows.append(
            {
                "model": name,
                "n": len(y),
                "accuracy": accuracy_score(y, predictions),
                "precision": precision_score(y, predictions, zero_division=0),
                "recall": recall_score(y, predictions, zero_division=0),
                "f1": f1_score(y, predictions, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y, predictions),
                "roc_auc": roc_auc_score(y, probabilities),
            }
        )

        fpr, tpr, _ = roc_curve(y, probabilities)
        roc[name] = (fpr, tpr)
        oof_probabilities[name] = probabilities

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False), roc, oof_probabilities


def evaluate_policy_oof(X, y, groups, df, model):
    records = []

    for fold, (train_idx, test_idx) in enumerate(
        CV.split(X, y, groups=groups), start=1
    ):
        fold_model = Pipeline(
            [
                ("preprocess", make_preprocessor(CATEGORICAL, NUMERIC)),
                ("model", clone(model)),
            ]
        )

        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])

        test = df.iloc[test_idx].copy()
        test["predicted_success_probability"] = fold_model.predict_proba(
            test[FEATURES]
        )[:, 1]

        for gid, group in test.groupby(GROUP):
            if group["candidate_action"].nunique() != 5:
                continue

            chosen = group.loc[
                group["predicted_success_probability"].idxmax()
            ]

            records.append(
                {
                    "fold": fold,
                    "incident_group_id": gid,
                    "fault_type": chosen["fault_type"],
                    "severity": chosen["severity"],
                    "workload_profile": chosen["workload_profile"],
                    "selected_action": chosen["candidate_action"],
                    "selected_probability": chosen[
                        "predicted_success_probability"
                    ],
                    "selected_actual_success": int(chosen[TARGET]),
                    "oracle_success": int(group[TARGET].max()),
                }
            )

    result = pd.DataFrame(records)

    summary = pd.DataFrame(
        [
            {
                "complete_incident_groups_evaluated": len(result),
                "ml_policy_success_rate": result[
                    "selected_actual_success"
                ].mean(),
                "empirical_oracle_success_rate": result[
                    "oracle_success"
                ].mean(),
                "policy_regret": (
                    result["oracle_success"].mean()
                    - result["selected_actual_success"].mean()
                ),
            }
        ]
    )

    return result, summary


df = pd.read_csv(DATA)
df.columns = [c.strip() for c in df.columns]

missing_required = [c for c in [TARGET, GROUP] + CATEGORICAL if c not in df.columns]
if missing_required:
    raise ValueError(
        "Missing required columns: " + ", ".join(missing_required)
    )

NUMERIC = [c for c in NUMERIC_CANDIDATES if c in df.columns]
FEATURES = CATEGORICAL + NUMERIC

ml = df[df[TARGET].notna() & df[GROUP].notna()].copy()
ml[TARGET] = ml[TARGET].astype(int)

X = ml[FEATURES]
y = ml[TARGET]
groups = ml[GROUP]

CV = GroupKFold(n_splits=5)

# Models intentionally kept close to the research baseline.
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    ),
}

comparison, roc_data, oof_probs = evaluate_models(
    X, y, groups, models, CV
)
comparison.to_csv(OUT / "model_comparison.csv", index=False)

# ROC curves
plt.figure(figsize=(8, 6))
for name, (fpr, tpr) in roc_data.items():
    auc = comparison.loc[
        comparison["model"] == name, "roc_auc"
    ].iloc[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("GroupKFold ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "roc_curves.png", dpi=220)
plt.close()

best_name = comparison.iloc[0]["model"]
best_estimator = models[best_name]

best_pipe = Pipeline(
    [
        ("preprocess", make_preprocessor(CATEGORICAL, NUMERIC)),
        ("model", best_estimator),
    ]
)
best_pipe.fit(X, y)
joblib.dump(best_pipe, OUT / "trained_best_model.joblib")

# OOF policy evaluation
policy, policy_summary = evaluate_policy_oof(
    X, y, groups, ml, best_estimator
)
policy.to_csv(OUT / "oof_policy_results.csv", index=False)
policy_summary.to_csv(OUT / "oof_policy_summary.csv", index=False)

selection_dist = (
    policy["selected_action"]
    .value_counts()
    .rename_axis("selected_action")
    .reset_index(name="count")
)
selection_dist["percentage"] = (
    selection_dist["count"] / len(policy) * 100
)
selection_dist.to_csv(
    OUT / "oof_selected_action_distribution.csv",
    index=False,
)

# Feature importance for tree models.
# Use transformed feature names so categorical levels are visible.
rf_pipe = Pipeline(
    [
        ("preprocess", make_preprocessor(CATEGORICAL, NUMERIC)),
        ("model", clone(models["Random Forest"])),
    ]
)
rf_pipe.fit(X, y)

transformed_names = rf_pipe.named_steps[
    "preprocess"
].get_feature_names_out()

rf_importance = pd.DataFrame(
    {
        "feature": transformed_names,
        "importance": rf_pipe.named_steps[
            "model"
        ].feature_importances_,
    }
).sort_values("importance", ascending=False)

rf_importance.to_csv(
    OUT / "feature_importance.csv",
    index=False,
)

top = rf_importance.head(20).sort_values("importance")
plt.figure(figsize=(9, 8))
plt.barh(top["feature"], top["importance"])
plt.xlabel("Importance")
plt.title("Random Forest — Top 20 Features")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=220)
plt.close()

# SHAP for Random Forest on transformed features.
# TreeExplainer works on the fitted RF estimator.
transformed_X = rf_pipe.named_steps["preprocess"].transform(X)
rf_model = rf_pipe.named_steps["model"]

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(transformed_X)

# SHAP API differs between versions; for binary classifiers it may return
# either a list [class0, class1] or a 3D array.
if isinstance(shap_values, list):
    shap_matrix = shap_values[1]
elif getattr(shap_values, "ndim", 0) == 3:
    shap_matrix = shap_values[:, :, 1]
else:
    shap_matrix = shap_values

mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
shap_importance = pd.DataFrame(
    {
        "feature": transformed_names,
        "mean_abs_shap": mean_abs_shap,
    }
).sort_values("mean_abs_shap", ascending=False)

shap_importance.to_csv(
    OUT / "shap_global_importance.csv",
    index=False,
)

plt.figure(figsize=(9, 8))
top_shap = shap_importance.head(20).sort_values("mean_abs_shap")
plt.barh(top_shap["feature"], top_shap["mean_abs_shap"])
plt.xlabel("Mean |SHAP value|")
plt.title("Global SHAP Feature Importance")
plt.tight_layout()
plt.savefig(OUT / "shap_summary.png", dpi=220)
plt.close()

# Confusion matrix for best model
best_prob = oof_probs[best_name]
best_pred = (best_prob >= 0.5).astype(int)
cm = confusion_matrix(y, best_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Failure", "Success"])
plt.yticks([0, 1], ["Failure", "Success"])
for (i, j), value in np.ndenumerate(cm):
    plt.text(j, i, str(value), ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix_best_model.png", dpi=220)
plt.close()

# Metadata for reproducibility
metadata = {
    "dataset": str(DATA.name),
    "n_records": int(len(ml)),
    "n_incident_groups": int(ml[GROUP].nunique()),
    "features": FEATURES,
    "categorical_features": CATEGORICAL,
    "numeric_features": NUMERIC,
    "target": TARGET,
    "group_column": GROUP,
    "cv": "GroupKFold(n_splits=5)",
    "best_model": best_name,
    "post_recovery_features_excluded": True,
}
(OUT / "model_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("=" * 70)
print("RECOVERY ML CONTAINER")
print("=" * 70)
print(f"Dataset: {DATA.name}")
print(f"Records: {len(ml)}")
print(f"Incident groups: {ml[GROUP].nunique()}")
print(f"Features: {len(FEATURES)}")
print()
print(comparison.to_string(index=False))
print()
print("BEST MODEL:", best_name)
print()
print("OUT-OF-FOLD POLICY:")
print(policy_summary.to_string(index=False))
print()
print("Results written to:", OUT)
