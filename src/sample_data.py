from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


PUBLIC_DATASET_REFERENCE = {
    "dataset_name": "AI4I 2020 Predictive Maintenance Dataset",
    "dataset_owner": "UCI Machine Learning Repository",
    "dataset_reference": "AI4I 2020 Predictive Maintenance Dataset",
    "dataset_url": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
    "dataset_note": (
        "This project uses a compact local maintenance telemetry sample inspired by public predictive-maintenance "
        "datasets, adapted to a control tower prioritization workflow with operational criticality signals."
    ),
}


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", suffix=".csv", delete=False, dir=path.parent, encoding="utf-8") as tmp_file:
        temp_path = Path(tmp_file.name)
    try:
        df.to_csv(temp_path, index=False)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", suffix=".json", delete=False, dir=path.parent, encoding="utf-8") as tmp_file:
        temp_path = Path(tmp_file.name)
    try:
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _generate_sample(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assets = [
        ("CT-01", "pump"),
        ("CT-02", "compressor"),
        ("CT-03", "generator"),
        ("CT-04", "pump"),
        ("CT-05", "compressor"),
        ("CT-06", "blender"),
        ("CT-07", "generator"),
        ("CT-08", "pump"),
        ("CT-09", "compressor"),
        ("CT-10", "hydration_unit"),
    ]
    observed_cycle_plan = [58, 64, 69, 74, 79, 84, 89, 93, 97, 100]
    rows: list[dict[str, object]] = []

    for (asset_id, asset_type), observed_cycles in zip(assets, observed_cycle_plan, strict=True):
        base_temperature = rng.uniform(72, 91)
        base_vibration = rng.uniform(0.9, 1.9)
        base_pressure = rng.uniform(220, 310)
        base_current = rng.uniform(78, 116)
        base_efficiency = rng.uniform(89, 97)
        base_throughput = rng.uniform(82, 108)
        base_noise = rng.uniform(0.8, 1.5)
        production_impact = int(rng.integers(2, 6))
        redundancy_factor = int(rng.integers(1, 4))
        safety_criticality = int(rng.integers(1, 6))

        for cycle in range(1, observed_cycles + 1):
            lifecycle = cycle / 100
            anomaly_event = rng.random() < (0.02 + 0.08 * lifecycle)

            temperature = base_temperature + lifecycle * rng.uniform(12, 30) + rng.normal(0, 0.9)
            vibration = base_vibration + lifecycle * rng.uniform(1.1, 2.6) + rng.normal(0, 0.05)
            pressure = base_pressure - lifecycle * rng.uniform(20, 58) + rng.normal(0, 2.1)
            current = base_current + lifecycle * rng.uniform(14, 34) + rng.normal(0, 1.1)
            efficiency = base_efficiency - lifecycle * rng.uniform(7, 16) + rng.normal(0, 0.45)
            throughput = base_throughput - lifecycle * rng.uniform(8, 22) + rng.normal(0, 0.7)
            noise_index = base_noise + lifecycle * rng.uniform(0.8, 2.4) + rng.normal(0, 0.05)

            risk_score = (
                0.18 * max(temperature - 98, 0) / 7
                + 0.20 * max(vibration - 2.5, 0)
                + 0.12 * max(236 - pressure, 0) / 12
                + 0.12 * max(current - 122, 0) / 10
                + 0.18 * max(88 - efficiency, 0) / 4
                + 0.10 * max(84 - throughput, 0) / 4
                + 0.10 * max(noise_index - 2.5, 0)
            )
            maintenance_required = int(anomaly_event or risk_score > 0.38 or lifecycle > 0.82)

            rows.append(
                {
                    "asset_id": asset_id,
                    "asset_type": asset_type,
                    "cycle": cycle,
                    "temperature": round(float(temperature), 2),
                    "vibration": round(float(vibration), 3),
                    "pressure": round(float(pressure), 2),
                    "current": round(float(current), 2),
                    "efficiency": round(float(efficiency), 2),
                    "throughput": round(float(throughput), 2),
                    "noise_index": round(float(noise_index), 3),
                    "production_impact": production_impact,
                    "redundancy_factor": redundancy_factor,
                    "safety_criticality": safety_criticality,
                    "maintenance_required": maintenance_required,
                }
            )

    return pd.DataFrame(rows)


def ensure_dataset(base_dir: str | Path) -> dict[str, str]:
    base_path = Path(base_dir)
    telemetry_path = base_path / "data" / "raw" / "maintenance_control_tower_sample.csv"
    reference_path = base_path / "data" / "raw" / "public_dataset_reference.json"

    telemetry_df = _generate_sample()
    _atomic_write_csv(telemetry_df, telemetry_path)
    _atomic_write_json(PUBLIC_DATASET_REFERENCE, reference_path)

    return {
        "telemetry_path": str(telemetry_path),
        "reference_path": str(reference_path),
    }
