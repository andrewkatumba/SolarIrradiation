#!/usr/bin/env python3
"""Cloud Run Job entrypoint for refreshing the GCS model-release registry index."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from model_release_registry import build_index, write_markdown_index


def gcloud_bin() -> str:
    return os.environ.get("GCLOUD_BIN", "gcloud")


def run_command(args: list[str]) -> str:
    completed = subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def gcs_uri(bucket: str, *parts: str) -> str:
    joined = "/".join(part.strip("/") for part in parts if part)
    return f"gs://{bucket}/{joined}"


def list_manifest_uris(bucket: str, model_family: str) -> list[str]:
    pattern = gcs_uri(bucket, model_family, "*", "manifest.json")
    output = run_command([gcloud_bin(), "storage", "ls", pattern])
    return [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith(f"gs://{bucket}/{model_family}/")
    ]


def release_id_from_manifest_uri(bucket: str, model_family: str, uri: str) -> str:
    prefix = f"gs://{bucket}/{model_family}/"
    if not uri.startswith(prefix):
        raise ValueError(f"Manifest URI is outside the expected prefix: {uri}")
    return uri[len(prefix) :].split("/", 1)[0]


def download_manifests(bucket: str, model_family: str, project_root: Path) -> list[str]:
    manifest_uris = list_manifest_uris(bucket, model_family)
    release_ids = []
    for uri in manifest_uris:
        release_id = release_id_from_manifest_uri(bucket, model_family, uri)
        if release_id.startswith("_"):
            continue
        destination = project_root / model_family / release_id / "manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        run_command([gcloud_bin(), "storage", "cp", uri, str(destination)])
        release_ids.append(release_id)
    return sorted(release_ids)


def upload_outputs(
    bucket: str,
    model_family: str,
    output_prefix: str,
    run_id: str,
    output_dir: Path,
) -> None:
    destinations = {
        output_dir / "index.json": [
            gcs_uri(bucket, model_family, output_prefix, "latest", "index.json"),
            gcs_uri(bucket, model_family, output_prefix, "runs", run_id, "index.json"),
        ],
        output_dir / "index.md": [
            gcs_uri(bucket, model_family, output_prefix, "latest", "index.md"),
            gcs_uri(bucket, model_family, output_prefix, "runs", run_id, "index.md"),
        ],
        output_dir / "summary.json": [
            gcs_uri(bucket, model_family, output_prefix, "latest", "summary.json"),
            gcs_uri(bucket, model_family, output_prefix, "runs", run_id, "summary.json"),
        ],
    }
    for source, uris in destinations.items():
        for uri in uris:
            run_command([gcloud_bin(), "storage", "cp", str(source), uri])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--model-family", default="ghi-rf")
    parser.add_argument("--output-prefix", default="_registry")
    args = parser.parse_args(argv or sys.argv[1:])

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        project_root = temp_root / "registry-root"
        output_dir = temp_root / "outputs"
        output_dir.mkdir()

        release_ids = download_manifests(args.bucket, args.model_family, project_root)
        results = build_index(project_root, verify_files=False, gate=None)
        payload = [result.as_dict(project_root) for result in results]
        summary = {
            "run_id": run_id,
            "bucket": args.bucket,
            "model_family": args.model_family,
            "release_count": len(release_ids),
            "manifest_count": len(results),
            "pass_count": sum(1 for result in results if result.passed),
            "fail_count": sum(1 for result in results if not result.passed),
            "release_ids": release_ids,
        }

        (output_dir / "index.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_markdown_index(results, output_dir / "index.md", project_root)
        upload_outputs(args.bucket, args.model_family, args.output_prefix, run_id, output_dir)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
