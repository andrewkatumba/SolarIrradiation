# Modeling Scripts

These scripts support reproducible model experiments and release-candidate packaging for the Solar Irradiation project.

Project experiment records live in:

`/Users/stimak/Library/Mobile Documents/com~apple~CloudDocs/SolarIrradiation/04_modeling_validation/experiments/`

Large generated artifacts live in:

`/Users/stimak/Library/Mobile Documents/com~apple~CloudDocs/SolarIrradiation/04_modeling_validation/model_releases/`

## Current Scripts

| Script | Purpose | Primary Experiment |
| --- | --- | --- |
| `release_rf_model.py` | Build a manifest-first Random Forest release candidate from a CSV training snapshot, including artifact hashes and validation summary. | `EXP-RF-001` / `ghi-rf-2026-06-05-rc1` |
| `model_release_registry.py` | Validate model-release manifests, enforce staging/production gates, and generate a local registry index. | `MAR-31` lightweight MLOps registry |

## Environment

Use the pinned model-release dependencies:

```bash
python -m pip install -r requirements-model-release.txt
```

The MAR-20 run used the portal virtualenv because it already had compatible scientific packages:

`/Users/stimak/Documents/dev/solar_rradiation/Irradiation_Portal/.venv/bin/python`

## Recreate EXP-RF-001

```bash
/Users/stimak/Documents/dev/solar_rradiation/Irradiation_Portal/.venv/bin/python scripts/release_rf_model.py \
  --release-id ghi-rf-2026-06-05-rc1 \
  --training-csv "/Users/stimak/Library/Mobile Documents/com~apple~CloudDocs/SolarIrradiation/jupyter_notebooks/Data/all_locations_train_data.csv" \
  --monthly-ground-truth-csv "/Users/stimak/Library/Mobile Documents/com~apple~CloudDocs/SolarIrradiation/jupyter_notebooks/monthly_ground_truth_df.csv" \
  --monthly-predictions-csv "/Users/stimak/Library/Mobile Documents/com~apple~CloudDocs/SolarIrradiation/jupyter_notebooks/monthly_predictions.csv" \
  --output-dir "/Users/stimak/Library/Mobile Documents/com~apple~CloudDocs/SolarIrradiation/04_modeling_validation/model_releases/ghi-rf"
```

The command creates a new release directory and refuses to overwrite an existing one. Remove or choose a new `--release-id` before rerunning.

## Adding Future Experiment Scripts

- Keep reusable Python runners in this `scripts/` directory.
- Keep experiment-specific results in the project `experiments/` registry and `model_releases/`.
- Name new scripts by action and model family, for example `compare_rf_splits.py` or `export_rf_feature_fixture.py`.
- Every script that produces model artifacts should write hashes and enough configuration to recreate the run.

## Validate a Release Manifest

Use the lightweight registry gate before copying a release into staging:

```bash
python scripts/model_release_registry.py validate \
  "/Users/stimak/Library/Mobile Documents/com~apple~CloudDocs/SolarIrradiation/04_modeling_validation/experiments/EXP-RF-005/serving_release/manifest.json" \
  --project-root "/Users/stimak/Library/Mobile Documents/com~apple~CloudDocs/SolarIrradiation" \
  --verify-files \
  --gate staging
```

Generate a full local index:

```bash
python scripts/model_release_registry.py index \
  --project-root "/Users/stimak/Library/Mobile Documents/com~apple~CloudDocs/SolarIrradiation" \
  --output-json "/Users/stimak/Library/Mobile Documents/com~apple~CloudDocs/SolarIrradiation/04_modeling_validation/model_release_registry_index.json" \
  --output-md "/Users/stimak/Library/Mobile Documents/com~apple~CloudDocs/SolarIrradiation/04_modeling_validation/model_release_registry_index.md"
```
