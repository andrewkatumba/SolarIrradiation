#!/usr/bin/env python3
"""Create a manifest-first Random Forest model release from a CSV snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DIFF",
    "ALLSKY_SFC_SW_DNI",
    "CLRSKY_SFC_LW_DWN",
    "T2M",
    "RH2M",
    "PS",
    "WS2M",
    "AOD_55_ADJ",
    "CLOUD_AMT",
    "PW",
    "T2M_RANGE",
    "PRECTOTCORR",
    "Z0M",
    "V2M",
    "AIRMASS",
    "DISPH",
    "EVPTRNS",
    "PBLTOP",
    "TO3",
    "RHOA",
    "SRF_ALB_ADJ",
    "EVLAND",
    "GWETTOP",
    "ALLSKY_KT",
    "ghi_clear_cams",
    "bhi_clear_cams",
    "dhi_clear_cams",
    "dni_clear_cams",
    "ghi_cams",
    "bhi_cams",
    "dhi_cams",
    "dni_cams",
    "kt_cams",
    "longitude",
    "day_of_year",
    "altitude",
    "dayofyear_sin",
    "dayofyear_cos",
]

TARGET_COLUMN = "ghi"
ESTIMATED_GHI_COLUMN = "ALLSKY_SFC_SW_DWN"
MODEL_PARAMS = {
    "n_estimators": 200,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(path: Path, role: str, source: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "role": role,
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if source is not None:
        entry["source"] = source
    return entry


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["datetime"] = pd.to_datetime(prepared["datetime"], errors="raise")

    if "day_of_year" not in prepared.columns:
        if "DoY" in prepared.columns:
            prepared["day_of_year"] = prepared["DoY"]
        else:
            prepared["day_of_year"] = prepared["datetime"].dt.dayofyear

    day_angle = 2 * np.pi * prepared["day_of_year"].astype(float) / 365.0
    prepared["dayofyear_sin"] = np.sin(day_angle)
    prepared["dayofyear_cos"] = np.cos(day_angle)

    if "kt_cams" not in prepared.columns:
        denominator = prepared["ghi_clear_cams"].replace(0, np.nan)
        prepared["kt_cams"] = prepared["ghi_cams"] / denominator

    return prepared


def regression_metrics(y_true: pd.Series, corrected: np.ndarray, baseline: pd.Series) -> dict[str, Any]:
    return {
        "model": {
            "rmse": float(np.sqrt(mean_squared_error(y_true, corrected))),
            "mae": float(mean_absolute_error(y_true, corrected)),
            "r2": float(r2_score(y_true, corrected)),
            "mbe": float(np.mean(corrected - y_true.to_numpy())),
        },
        "baseline": {
            "rmse": float(np.sqrt(mean_squared_error(y_true, baseline))),
            "mae": float(mean_absolute_error(y_true, baseline)),
            "r2": float(r2_score(y_true, baseline)),
            "mbe": float(np.mean(baseline.to_numpy() - y_true.to_numpy())),
        },
    }


def monthly_snapshot_metrics(ground_truth_csv: Path | None, predictions_csv: Path | None) -> dict[str, Any] | None:
    if ground_truth_csv is None or predictions_csv is None:
        return None

    ground = pd.read_csv(ground_truth_csv)
    pred = pd.read_csv(predictions_csv)
    numeric_month_cols = [str(i) for i in range(1, 13)]
    named_month_cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if all(col in ground.columns and col in pred.columns for col in named_month_cols):
        month_cols = named_month_cols
    else:
        month_cols = numeric_month_cols
    missing = [col for col in month_cols if col not in ground.columns or col not in pred.columns]
    if missing:
        raise ValueError(f"Monthly snapshot files are missing month columns: {missing}")

    y_true = ground[month_cols].to_numpy(dtype=float).ravel()
    y_pred = pred[month_cols].to_numpy(dtype=float).ravel()
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        raise ValueError("Monthly snapshot files contain no finite prediction/ground-truth pairs")

    return {
        "source": "snapshot_recompute_not_generated_by_this_run",
        "pair_count": int(valid.sum()),
        "location_count": int(len(ground)),
        "month_columns": month_cols,
        "prediction_vs_ground": {
            "rmse": float(np.sqrt(mean_squared_error(y_true[valid], y_pred[valid]))),
            "mae": float(mean_absolute_error(y_true[valid], y_pred[valid])),
            "r2": float(r2_score(y_true[valid], y_pred[valid])),
            "mbe": float(np.mean(y_pred[valid] - y_true[valid])),
        },
    }


def write_validation_report(
    path: Path,
    release_id: str,
    holdout_metrics: dict[str, Any],
    monthly_metrics: dict[str, Any] | None,
) -> None:
    lines = [
        f"# Validation Report: {release_id}",
        "",
        "Status: release-candidate evidence only; not a production approval.",
        "",
        "## Internal Holdout",
        "",
        "| Metric | Model corrected GHI | Baseline NASA POWER GHI |",
        "| --- | ---: | ---: |",
    ]
    for metric in ["rmse", "mae", "r2", "mbe"]:
        lines.append(
            f"| {metric.upper()} | {holdout_metrics['model'][metric]:.4f} | {holdout_metrics['baseline'][metric]:.4f} |"
        )
    lines.extend(
        [
            "",
            "The internal holdout uses the final chronological split from the candidate training snapshot.",
            "The exported model is retrained on all clean rows after this diagnostic holdout is measured.",
        ]
    )
    if monthly_metrics is not None:
        snap = monthly_metrics["prediction_vs_ground"]
        lines.extend(
            [
                "",
                "## Monthly Snapshot Recompute",
                "",
                "These monthly metrics are recomputed from existing notebook snapshot CSVs and were not generated by this retraining run.",
                "",
                "| Metric | Prediction vs ground |",
                "| --- | ---: |",
                f"| RMSE | {snap['rmse']:.4f} |",
                f"| MAE | {snap['mae']:.4f} |",
                f"| R2 | {snap['r2']:.4f} |",
                f"| MBE | {snap['mbe']:.4f} |",
                f"| Pairs | {monthly_metrics['pair_count']} |",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def build_release(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve() / args.release_id
    output_dir.mkdir(parents=True, exist_ok=False)

    training_csv = Path(args.training_csv).expanduser().resolve()
    raw = pd.read_csv(training_csv)
    prepared = add_derived_features(raw)

    required = [TARGET_COLUMN, ESTIMATED_GHI_COLUMN, *FEATURE_COLUMNS]
    missing = [col for col in required if col not in prepared.columns]
    if missing:
        raise ValueError(f"Training snapshot is missing required columns: {missing}")

    clean = prepared.dropna(subset=required).sort_values("datetime").reset_index(drop=True)
    if len(clean) < 10:
        raise ValueError(f"Training snapshot has too few clean rows: {len(clean)}")

    target = clean[TARGET_COLUMN] - clean[ESTIMATED_GHI_COLUMN]
    split_at = int(len(clean) * (1 - args.holdout_fraction))
    if split_at <= 0 or split_at >= len(clean):
        raise ValueError("Holdout fraction leaves an empty train or validation partition")

    scaler_for_holdout = StandardScaler()
    x_train = scaler_for_holdout.fit_transform(clean.loc[: split_at - 1, FEATURE_COLUMNS])
    x_valid = scaler_for_holdout.transform(clean.loc[split_at:, FEATURE_COLUMNS])
    y_train = target.iloc[:split_at]
    y_valid = clean.loc[split_at:, TARGET_COLUMN]
    baseline_valid = clean.loc[split_at:, ESTIMATED_GHI_COLUMN]

    holdout_model = RandomForestRegressor(**MODEL_PARAMS)
    holdout_model.fit(x_train, y_train)
    corrected_valid = baseline_valid.to_numpy() + holdout_model.predict(x_valid)
    holdout_metrics = regression_metrics(y_valid, corrected_valid, baseline_valid)

    final_scaler = StandardScaler()
    x_all = final_scaler.fit_transform(clean[FEATURE_COLUMNS])
    final_model = RandomForestRegressor(**MODEL_PARAMS)
    final_model.fit(x_all, target)

    metadata = {
        "feature_cols": FEATURE_COLUMNS,
        "ground_truth_col": TARGET_COLUMN,
        "est_ghi_col": ESTIMATED_GHI_COLUMN,
        "model_type": "random_forest",
        "model_params": {},
        "best_params_": {},
        "scaler_used": True,
        "target_mode": "bias_ground_minus_satellite",
        "final_output": "nasa_ghi + predicted_bias",
        "unit": "kWh/m2/day",
        "model_release_id": args.release_id,
        "model_family": "ghi-rf",
        "production_approval": "not_approved",
    }

    model_path = output_dir / "random_forest_model.joblib"
    scaler_path = output_dir / "random_forest_scaler.joblib"
    metadata_path = output_dir / "random_forest_metadata.joblib"
    joblib.dump(final_model, model_path)
    joblib.dump(final_scaler, scaler_path)
    joblib.dump(metadata, metadata_path)

    source_training_copy = output_dir / "training_snapshot.csv"
    shutil.copy2(training_csv, source_training_copy)

    monthly_metrics = monthly_snapshot_metrics(args.monthly_ground_truth_csv, args.monthly_predictions_csv)
    validation_report_path = output_dir / "validation_report.md"
    write_validation_report(validation_report_path, args.release_id, holdout_metrics, monthly_metrics)

    release_files = {
        "model": file_entry(model_path, "random_forest_model"),
        "scaler": file_entry(scaler_path, "standard_scaler"),
        "metadata": file_entry(metadata_path, "model_metadata"),
        "training_snapshot": file_entry(
            source_training_copy,
            "training_data_snapshot",
            source=str(training_csv),
        ),
        "validation_report": file_entry(validation_report_path, "validation_report"),
    }

    validation_snapshots: dict[str, Any] = {}
    if args.monthly_ground_truth_csv is not None:
        ground_copy = output_dir / "monthly_ground_truth_df.csv"
        shutil.copy2(args.monthly_ground_truth_csv, ground_copy)
        validation_snapshots["monthly_ground_truth"] = file_entry(
            ground_copy,
            "monthly_validation_ground_truth_snapshot",
            source=str(Path(args.monthly_ground_truth_csv).expanduser().resolve()),
        )
    if args.monthly_predictions_csv is not None:
        pred_copy = output_dir / "monthly_predictions.csv"
        shutil.copy2(args.monthly_predictions_csv, pred_copy)
        validation_snapshots["monthly_predictions"] = file_entry(
            pred_copy,
            "monthly_validation_prediction_snapshot",
            source=str(Path(args.monthly_predictions_csv).expanduser().resolve()),
        )

    manifest = {
        "manifest_schema": "solar-irradiance-model-release/v1",
        "model_release_id": args.release_id,
        "model_family": "ghi-rf",
        "status": "release_candidate_staging_only",
        "deployment_status": "not_deployed",
        "production_approval": "blocked_pending_owner_modeling_approval",
        "new_rf_model_used": False,
        "target_mode": "bias_ground_minus_satellite",
        "final_output": "nasa_ghi + predicted_bias",
        "unit": "kWh/m2/day",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "owner": args.owner,
        "source_recovery": {
            "deployed_model_sha256": args.deployed_model_sha256,
            "exact_deployed_binary_recovered": False,
            "recovery_result": "exact hash not found outside the portal RF_MODEL artifact during MAR-20 local search",
        },
        "training": {
            "repo": "SolarIrradiation",
            "repo_commit": git_commit(Path(__file__).resolve().parents[1]),
            "command": " ".join(args.command_argv),
            "input_rows": int(len(raw)),
            "clean_rows": int(len(clean)),
            "date_range": {
                "start": clean["datetime"].min().date().isoformat(),
                "end": clean["datetime"].max().date().isoformat(),
            },
            "location_count": int(clean["location"].nunique()) if "location" in clean else None,
            "target": f"{TARGET_COLUMN} - {ESTIMATED_GHI_COLUMN}",
            "holdout_fraction": args.holdout_fraction,
            "final_export": "trained_on_all_clean_rows_after_holdout_diagnostic",
        },
        "feature_schema": {
            "count": len(FEATURE_COLUMNS),
            "columns": FEATURE_COLUMNS,
            "derived_features": {
                "day_of_year": "DoY column when present, otherwise datetime.dayofyear",
                "dayofyear_sin": "sin(2*pi*day_of_year/365)",
                "dayofyear_cos": "cos(2*pi*day_of_year/365)",
                "kt_cams": "ghi_cams / ghi_clear_cams",
            },
        },
        "model_parameters": MODEL_PARAMS,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "artifacts": release_files,
        "validation_snapshots": validation_snapshots,
        "metrics": {
            "internal_holdout": holdout_metrics,
            "monthly_snapshot_recompute": monthly_metrics,
        },
        "serving_contract": {
            "select_by": "MODEL_RELEASE_ID",
            "recommended_storage": "private GCS bucket such as gs://solar-498315-model-releases/ghi-rf/<MODEL_RELEASE_ID>/",
            "image_build_contract": "copy exactly one approved release into the image by MODEL_RELEASE_ID and verify manifest hashes before startup",
            "vertex_ai_model_registry": "defer until managed model governance or evaluation workflow is needed",
        },
        "rollback": {
            "restore_model_release_id": "previous approved MODEL_RELEASE_ID",
            "verify": [
                "manifest.json SHA-256 matches the expected release record",
                "artifact file SHA-256 values match manifest entries",
                "offline prediction fixture or smoke test passes before traffic shift",
            ],
        },
        "limitations": [
            "This release does not recover the exact deployed model binary.",
            "Monthly validation snapshot metrics are inherited from notebook CSVs and are not generated by this retrained model.",
            "Production promotion remains blocked until owner/modeling approval is recorded.",
            "New_RF_model is intentionally excluded.",
        ],
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True, help="MODEL_RELEASE_ID, for example ghi-rf-2026-06-05-rc1")
    parser.add_argument("--training-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--monthly-ground-truth-csv", type=Path)
    parser.add_argument("--monthly-predictions-csv", type=Path)
    parser.add_argument("--owner", default="Andrew Katumba / Marconi Lab modeling owner")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument(
        "--deployed-model-sha256",
        default="c477c162e86589057be224fc9c27f3c069885f9909b00cf000c7bf302571f741",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.command_argv = sys.argv
    result = build_release(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
