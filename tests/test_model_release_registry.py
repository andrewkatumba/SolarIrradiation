from __future__ import annotations

import json
from pathlib import Path

from scripts.model_release_registry import build_index, validate_manifest


def write_release(tmp_path: Path, **overrides):
    release = tmp_path / "EXP-RF-TEST"
    release.mkdir()
    model = release / "random_forest_model.joblib"
    metadata = release / "random_forest_metadata.joblib"
    model.write_bytes(b"model")
    metadata.write_bytes(b"metadata")
    manifest = {
        "model_release_id": "EXP-RF-TEST",
        "model_family": "ghi-rf",
        "status": "staging-review",
        "deployment_status": "staging_review",
        "production_approval": "not_approved",
        "new_rf_model_used": False,
        "target_mode": "direct_ghi",
        "unit": "kWh/m2/day",
        "artifacts": [
            {
                "role": "random_forest_model",
                "path": "random_forest_model.joblib",
                "sha256": "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4",
            },
            {
                "role": "model_metadata",
                "path": "random_forest_metadata.joblib",
                "sha256": "45447b7afbd5e544f7d0f1df0fccd26014d9850130abd3f020b89ff96b82079f",
            },
        ],
    }
    manifest.update(overrides)
    manifest_path = release / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_validate_staging_release_with_artifact_hashes(tmp_path):
    manifest_path = write_release(tmp_path)

    result = validate_manifest(
        manifest_path,
        project_root=tmp_path,
        verify_files=True,
        gate="staging",
    )

    assert result.passed
    assert result.release_id == "EXP-RF-TEST"
    assert result.verified_artifacts == 2


def test_validate_blocks_new_rf_model(tmp_path):
    manifest_path = write_release(tmp_path, new_rf_model_used=True)

    result = validate_manifest(manifest_path, project_root=tmp_path, gate="staging")

    assert not result.passed
    assert any("new_rf_model_used=true" in error for error in result.errors)


def test_validate_staging_gate_blocks_unsupported_target_mode(tmp_path):
    manifest_path = write_release(tmp_path, target_mode="bias_ground_minus_cams")

    result = validate_manifest(manifest_path, project_root=tmp_path, gate="staging")

    assert not result.passed
    assert any("not supported by portal serving" in error for error in result.errors)


def test_build_index_finds_manifest(tmp_path):
    write_release(tmp_path)

    results = build_index(tmp_path, verify_files=False, gate="staging")

    assert [result.release_id for result in results] == ["EXP-RF-TEST"]
    assert results[0].passed
