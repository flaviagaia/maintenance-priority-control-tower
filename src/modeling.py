from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.sample_data import ensure_dataset


def _priority_band(priority_score: float) -> str:
    if priority_score >= 80:
        return "P1"
    if priority_score >= 52:
        return "P2"
    if priority_score >= 35:
        return "P3"
    return "P4"


def run_pipeline(base_dir: str | Path) -> dict:
    base_path = Path(base_dir)
    dataset = ensure_dataset(base_path)
    telemetry = pd.read_csv(dataset["telemetry_path"])

    feature_columns = [
        "asset_id",
        "asset_type",
        "cycle",
        "temperature",
        "vibration",
        "pressure",
        "current",
        "efficiency",
        "throughput",
        "noise_index",
        "production_impact",
        "redundancy_factor",
        "safety_criticality",
    ]
    X = telemetry[feature_columns]
    y = telemetry["maintenance_required"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    numeric_features = [
        "cycle",
        "temperature",
        "vibration",
        "pressure",
        "current",
        "efficiency",
        "throughput",
        "noise_index",
        "production_impact",
        "redundancy_factor",
        "safety_criticality",
    ]
    categorical_features = ["asset_id", "asset_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=320,
                    max_depth=10,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.42).astype(int)

    roc_auc = float(roc_auc_score(y_test, probabilities))
    average_precision = float(average_precision_score(y_test, probabilities))
    f1 = float(f1_score(y_test, predictions))

    processed_dir = base_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    scored_path = processed_dir / "maintenance_scored_cycles.csv"
    tower_path = processed_dir / "maintenance_priority_tower.csv"
    report_path = processed_dir / "maintenance_priority_control_tower_report.json"

    scored_df = X_test.copy()
    scored_df["maintenance_required"] = y_test.values
    scored_df["predicted_probability"] = probabilities.round(4)
    scored_df["predicted_label"] = predictions
    scored_df.to_csv(scored_path, index=False)

    latest_cycles = telemetry.sort_values(["asset_id", "cycle"]).groupby("asset_id", as_index=False).tail(1).copy()
    latest_features = latest_cycles[feature_columns]
    latest_probabilities = model.predict_proba(latest_features)[:, 1]
    latest_cycles["predicted_probability"] = latest_probabilities.round(4)
    latest_cycles["risk_score"] = (latest_probabilities * 100).round(2)
    latest_cycles["priority_score"] = (
        latest_probabilities * 48
        + latest_cycles["production_impact"] * 5.5
        + latest_cycles["safety_criticality"] * 5
        + (4 - latest_cycles["redundancy_factor"]) * 4.5
    ).round(2)
    latest_cycles["priority_band"] = latest_cycles["priority_score"].apply(_priority_band)
    latest_cycles["recommended_action"] = latest_cycles["priority_band"].map(
        {
            "P1": "dispatch_immediate_maintenance",
            "P2": "schedule_next_shift",
            "P3": "monitor_and_prepare_parts",
            "P4": "routine_observation",
        }
    )
    latest_cycles = latest_cycles.sort_values(
        ["priority_score", "predicted_probability", "safety_criticality"],
        ascending=[False, False, False],
    )
    latest_cycles.to_csv(tower_path, index=False)

    summary = {
        "dataset_source": "maintenance_control_tower_ai4i_style",
        "public_dataset_reference": dataset["reference_path"],
        "row_count": int(len(telemetry)),
        "asset_count": int(telemetry["asset_id"].nunique()),
        "positive_rate": round(float(telemetry["maintenance_required"].mean()), 4),
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(average_precision, 4),
        "f1": round(f1, 4),
        "p1_assets": int((latest_cycles["priority_band"] == "P1").sum()),
        "control_tower_artifact": str(tower_path),
        "scored_artifact": str(scored_path),
        "report_artifact": str(report_path),
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
