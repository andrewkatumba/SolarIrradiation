#!/usr/bin/env python3
"""Validate and index Solar Irradiance model-release manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


STATUS_ALIASES = {
    "experimental": "experiment",
    "research_only": "experiment",
    "release_candidate_staging_only": "candidate",
    "staging_review": "staging-review",
    "staging-review": "staging-review",
    "production_approved": "production-approved",
    "production-approved": "production-approved",
}
CANONICAL_STATUSES = {
    "experiment",
    "candidate",
    "staging-review",
    "production-approved",
    "retired",
}
SERVING_TARGET_MODES = {
    "direct_ghi",
    "bias_ground_minus_satellite",
    "bias",
    "bias_satellite_minus_ground",
}
REQUIRED_SERVING_FIELDS = {
    "model_release_id",
    "model_family",
    "status",
    "deployment_status",
    "production_approval",
    "target_mode",
    "unit",
}


@dataclass
class ValidationResult:
    """Result of a manifest validation pass."""

    manifest_path: Path
    release_id: str
    status: str
    deployment_status: str
    production_approval: str
    target_mode: str
    gate: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verified_artifacts: int = 0

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self, project_root: Path | None = None) -> dict[str, Any]:
        manifest_path = self.manifest_path
        if project_root is not None:
            try:
                manifest_path = manifest_path.relative_to(project_root)
            except ValueError:
                pass
        return {
            "release_id": self.release_id,
            "manifest": str(manifest_path),
            "status": self.status,
            "deployment_status": self.deployment_status,
            "production_approval": self.production_approval,
            "target_mode": self.target_mode,
            "gate": self.gate,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "verified_artifacts": self.verified_artifacts,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {manifest_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
    return data


def release_id(manifest: dict[str, Any]) -> str:
    return str(
        manifest.get("model_release_id")
        or manifest.get("experiment_id")
        or manifest.get("release_id")
        or "UNKNOWN"
    )


def normalized_status(manifest: dict[str, Any]) -> str:
    raw = str(manifest.get("status", "missing")).strip()
    return STATUS_ALIASES.get(raw, raw)


def artifact_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw_artifacts = manifest.get("artifacts", [])
    entries: list[dict[str, Any]] = []
    if isinstance(raw_artifacts, list):
        for item in raw_artifacts:
            if isinstance(item, dict):
                entries.append(dict(item))
    elif isinstance(raw_artifacts, dict):
        for role, item in raw_artifacts.items():
            if isinstance(item, dict):
                entry = dict(item)
                entry.setdefault("role", role)
                entries.append(entry)
    return entries


def artifact_roles(manifest: dict[str, Any]) -> set[str]:
    return {str(entry.get("role", "")) for entry in artifact_entries(manifest)}


def resolve_artifact_path(
    raw_path: str,
    manifest_dir: Path,
    project_root: Path | None,
) -> Path:
    artifact_path = Path(raw_path)
    if artifact_path.is_absolute():
        return artifact_path

    candidates = [manifest_dir / artifact_path]
    if project_root is not None:
        candidates.append(project_root / artifact_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_manifest(
    manifest_path: Path,
    project_root: Path | None = None,
    verify_files: bool = False,
    gate: str | None = None,
) -> ValidationResult:
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    result = ValidationResult(
        manifest_path=manifest_path,
        release_id=release_id(manifest),
        status=normalized_status(manifest),
        deployment_status=str(manifest.get("deployment_status", "missing")),
        production_approval=str(manifest.get("production_approval", "missing")),
        target_mode=str(manifest.get("target_mode", "missing")),
        gate=gate,
    )

    if result.status not in CANONICAL_STATUSES:
        result.warnings.append(
            f"Status {manifest.get('status')!r} is not a canonical registry status"
        )

    if manifest.get("new_rf_model_used") is True:
        result.errors.append("new_rf_model_used=true is blocked by project guardrails")

    is_serving_release = "model_release_id" in manifest
    if is_serving_release:
        missing_fields = [
            field for field in sorted(REQUIRED_SERVING_FIELDS) if field not in manifest
        ]
        for field_name in missing_fields:
            result.errors.append(f"Missing required serving field: {field_name}")

        roles = artifact_roles(manifest)
        has_model = bool(roles & {"random_forest_model", "candidate_artifact"})
        has_metadata = bool(roles & {"model_metadata", "candidate_metadata"})
        if not has_model:
            result.errors.append(
                "Manifest must identify a random_forest_model or candidate_artifact"
            )
        if not has_metadata:
            result.errors.append(
                "Manifest must identify model_metadata or candidate_metadata"
            )
    else:
        result.warnings.append("Manifest is experiment-only and is not deployable")

    if gate:
        validate_gate(manifest, result, gate)

    if verify_files:
        verify_artifacts(manifest, result, project_root)

    return result


def validate_gate(
    manifest: dict[str, Any],
    result: ValidationResult,
    gate: str,
) -> None:
    if gate not in {"staging", "production"}:
        result.errors.append(f"Unsupported gate: {gate}")
        return

    if "model_release_id" not in manifest:
        result.errors.append(f"{gate} gate requires model_release_id")

    if result.status == "retired":
        result.errors.append(f"{gate} gate cannot deploy a retired release")

    if result.target_mode not in SERVING_TARGET_MODES:
        result.errors.append(
            f"{gate} gate target_mode {result.target_mode!r} is not supported by portal serving"
        )

    if gate == "production":
        if result.status != "production-approved":
            result.errors.append("Production gate requires status=production-approved")
        if result.production_approval not in {"approved", "production_approved"}:
            result.errors.append("Production gate requires explicit production approval")
        if not manifest.get("approval"):
            result.errors.append("Production gate requires an approval record")
        if not manifest.get("rollback"):
            result.errors.append("Production gate requires rollback evidence")


def verify_artifacts(
    manifest: dict[str, Any],
    result: ValidationResult,
    project_root: Path | None,
) -> None:
    for artifact in artifact_entries(manifest):
        raw_path = artifact.get("path")
        if not raw_path:
            result.errors.append(f"Artifact entry missing path: {artifact}")
            continue
        path = resolve_artifact_path(
            str(raw_path),
            result.manifest_path.parent,
            project_root,
        )
        if not path.exists():
            result.errors.append(f"Artifact file does not exist: {raw_path}")
            continue
        expected_sha = artifact.get("sha256")
        if expected_sha:
            actual_sha = sha256_file(path)
            if actual_sha != expected_sha:
                result.errors.append(
                    f"SHA-256 mismatch for {raw_path}: expected {expected_sha}, got {actual_sha}"
                )
                continue
        result.verified_artifacts += 1


def discover_manifests(project_root: Path) -> list[Path]:
    ignored_parts = {".git", ".venv", "__pycache__", ".pytest_cache"}
    manifests = []
    for path in project_root.rglob("manifest.json"):
        if ignored_parts & set(path.parts):
            continue
        manifests.append(path)
    return sorted(manifests)


def build_index(
    project_root: Path,
    verify_files: bool = False,
    gate: str | None = None,
) -> list[ValidationResult]:
    return [
        validate_manifest(path, project_root, verify_files=verify_files, gate=gate)
        for path in discover_manifests(project_root)
    ]


def write_markdown_index(results: list[ValidationResult], path: Path, project_root: Path) -> None:
    lines = [
        "# Model Release Registry Index",
        "",
        "Generated by `scripts/model_release_registry.py`.",
        "",
        "| Release | Status | Deployment | Production | Target mode | Gate | Result | Manifest |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        try:
            manifest_path = result.manifest_path.relative_to(project_root)
        except ValueError:
            manifest_path = result.manifest_path
        gate = result.gate or "-"
        outcome = "pass" if result.passed else "fail"
        lines.append(
            f"| `{result.release_id}` | `{result.status}` | `{result.deployment_status}` | "
            f"`{result.production_approval}` | `{result.target_mode}` | `{gate}` | "
            f"`{outcome}` | `{manifest_path}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_result(result: ValidationResult, project_root: Path | None = None) -> None:
    print(json.dumps(result.as_dict(project_root), indent=2, sort_keys=True))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("manifest", type=Path)
    validate_parser.add_argument("--project-root", type=Path)
    validate_parser.add_argument("--verify-files", action="store_true")
    validate_parser.add_argument("--gate", choices=["staging", "production"])

    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--project-root", required=True, type=Path)
    index_parser.add_argument("--verify-files", action="store_true")
    index_parser.add_argument("--gate", choices=["staging", "production"])
    index_parser.add_argument("--output-json", type=Path)
    index_parser.add_argument("--output-md", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.command == "validate":
        project_root = args.project_root.resolve() if args.project_root else None
        result = validate_manifest(
            args.manifest,
            project_root,
            verify_files=args.verify_files,
            gate=args.gate,
        )
        print_result(result, project_root)
        return 0 if result.passed else 1

    project_root = args.project_root.resolve()
    results = build_index(project_root, verify_files=args.verify_files, gate=args.gate)
    payload = [result.as_dict(project_root) for result in results]
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown_index(results, args.output_md, project_root)
    if not args.output_json and not args.output_md:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
