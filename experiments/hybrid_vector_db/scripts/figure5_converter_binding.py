from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


class ConverterBindingError(RuntimeError):
    """A formal converter cannot close its provenance chain."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audited_run_manifest(
    path: Path,
    *,
    expected_artifact_type: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    resolved = path.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConverterBindingError(
            f"cannot read audited run manifest {resolved}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConverterBindingError("audited run manifest must be a JSON object")
    if payload.get("artifact_type") != expected_artifact_type:
        raise ConverterBindingError(
            f"unexpected audited artifact type: {payload.get('artifact_type')!r}"
        )
    if payload.get("status") != "complete":
        raise ConverterBindingError("audited run manifest is not complete")
    for field in (
        "artifact_valid",
        "requested_slice_complete",
        "full_release_complete",
        "paper_eligible",
    ):
        if payload.get(field) is not True:
            raise ConverterBindingError(
                f"audited run manifest failed {field}"
            )
    release = payload.get("release_contract")
    if not isinstance(release, Mapping):
        raise ConverterBindingError(
            "audited run manifest has no release contract"
        )
    required = (
        "path",
        "sha256",
        "contract_id",
        "expected_sqlens_build_id",
        "expected_vector_so_sha256",
    )
    normalized = {field: str(release.get(field) or "") for field in required}
    if any(not normalized[field] for field in required):
        raise ConverterBindingError(
            "audited run manifest has an incomplete release contract"
        )
    contract_path = Path(normalized["path"]).resolve()
    if (
        not contract_path.is_file()
        or sha256_file(contract_path) != normalized["sha256"]
    ):
        raise ConverterBindingError(
            "release contract path/SHA binding is invalid"
        )
    normalized["path"] = str(contract_path)
    return payload, normalized, sha256_file(resolved)


def row_provenance(
    release: Mapping[str, Any],
    source_manifest: Path,
    source_sha256: str,
) -> dict[str, str]:
    return {
        "release_contract_path": str(release["path"]),
        "release_contract_sha256": str(release["sha256"]),
        "release_contract_id": str(release["contract_id"]),
        "release_build_id": str(release["expected_sqlens_build_id"]),
        "release_vector_so_sha256": str(
            release["expected_vector_so_sha256"]
        ),
        "source_manifest_path": str(source_manifest.resolve()),
        "source_manifest_sha256": source_sha256,
    }


def publish_converter_binding(
    *,
    kind: str,
    source_manifest: Path,
    source_sha256: str,
    release: Mapping[str, Any],
    output: Path,
    rows: int,
    converter_source: Path,
    binding_path: Path,
) -> dict[str, Any]:
    resolved_output = output.resolve()
    resolved_source = source_manifest.resolve()
    if sha256_file(resolved_source) != source_sha256:
        raise ConverterBindingError(
            "audited run manifest changed during conversion"
        )
    if not resolved_output.is_file():
        raise ConverterBindingError(
            f"converter output does not exist: {resolved_output}"
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "sqlens_figure5_converter_binding",
        "experiment_kind": kind,
        "status": "complete",
        "artifact_valid": True,
        "requested_slice_complete": True,
        "full_release_complete": True,
        "paper_eligible": True,
        "created_at": utc_now(),
        "release_contract": dict(release),
        "execution_source": {
            "path": str(converter_source.resolve()),
            "sha256": sha256_file(converter_source.resolve()),
        },
        "converter_binding": {
            "source_manifest": {
                "path": str(resolved_source),
                "sha256": source_sha256,
            },
            "output": {
                "path": str(resolved_output),
                "sha256": sha256_file(resolved_output),
                "rows": rows,
                "experiment_kind": kind,
            },
        },
    }
    atomic_json(binding_path.resolve(), payload)
    return payload
