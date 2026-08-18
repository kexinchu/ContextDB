#!/usr/bin/env python3
"""Run the pinned Weaviate service used by the formal Amazon-10M experiment.

The controller is deliberately fail-closed: the image cannot be overridden,
startup requires absolute and percentage-based disk headroom, and a service is
only declared running after Docker inspection and ``/v1/meta`` both match the
recorded configuration.  Stopping removes the container but never the bind-
mounted corpus data.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
WEAVIATE_VERSION = "1.38.0"
IMAGE_REPOSITORY = "cr.weaviate.io/semitechnologies/weaviate"
IMAGE_DIGEST = "sha256:5ec2f15768eb59d9f5ea21edb29b6395d9844d9641caa694e33e8689f65fee0f"
IMAGE_REFERENCE = f"{IMAGE_REPOSITORY}@{IMAGE_DIGEST}"
CONTAINER_NAME = "weaviate-amazon10m-formal"
CLUSTER_HOSTNAME = "weaviate-amazon10m-formal-node"
SERVICE_LABEL_KEY = "com.hybrid-retrieval.service"
SERVICE_LABEL_VALUE = "weaviate-amazon10m-formal"
DATA_MOUNT_TARGET = "/var/lib/weaviate"
DEFAULT_DATA_DIR = ROOT / "data/weaviate-amazon10m-formal"
DEFAULT_MANIFEST = (
    ROOT
    / "results/hybrid_vector_db/weaviate_amazon10m_formal_service_runtime.json"
)
DEFAULT_BIND_ADDRESS = "127.0.0.1"
DEFAULT_REST_PORT = 8080
DEFAULT_GRPC_PORT = 50051
CONTAINER_REST_PORT = 8080
CONTAINER_GRPC_PORT = 50051
DEFAULT_CPUS = "32"
DEFAULT_MEMORY = "256g"
DEFAULT_DISK_USE_READONLY_PERCENTAGE = 98
DEFAULT_MIN_FREE_BYTES = 100 * 1024**3
WEAVIATE_DEFAULT_DISK_USE_READONLY_PERCENTAGE = 90
MANIFEST_SCHEMA_VERSION = 1
DOCKER_PLATFORM = "linux/amd64"

_BYTE_SIZE_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?)(?:i?b)?$", re.I)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ServiceError(RuntimeError):
    """The formal service failed a lifecycle or identity gate."""


@dataclass(frozen=True)
class ServiceConfig:
    container_name: str
    cluster_hostname: str
    data_dir: Path
    manifest: Path
    bind_address: str
    rest_port: int
    grpc_port: int
    cpus: str
    nano_cpus: int
    memory_bytes: int
    disk_use_readonly_percentage: int
    min_free_bytes: int
    startup_timeout: float
    poll_interval: float
    http_timeout: float

    def manifest_configuration(self) -> dict[str, Any]:
        return {
            "weaviate_version": WEAVIATE_VERSION,
            "image_reference": IMAGE_REFERENCE,
            "image_digest": IMAGE_DIGEST,
            "docker_platform": DOCKER_PLATFORM,
            "container_name": self.container_name,
            "cluster_hostname": self.cluster_hostname,
            "data_directory": str(self.data_dir),
            "data_mount_target": DATA_MOUNT_TARGET,
            "bind_address": self.bind_address,
            "rest_host_port": self.rest_port,
            "rest_container_port": CONTAINER_REST_PORT,
            "grpc_host_port": self.grpc_port,
            "grpc_container_port": CONTAINER_GRPC_PORT,
            "cpus": self.cpus,
            "nano_cpus": self.nano_cpus,
            "memory_bytes": self.memory_bytes,
            "disk_use_readonly_percentage": self.disk_use_readonly_percentage,
            "minimum_free_bytes_before_start": self.min_free_bytes,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    previous = path.read_bytes() if path.exists() else None
    replaced = False
    try:
        with temporary.open("w", encoding="utf-8") as target:
            json.dump(payload, target, sort_keys=True, indent=2, ensure_ascii=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        replaced = True
        fsync_directory(path.parent)
    except Exception:
        if replaced:
            try:
                if previous is None:
                    path.unlink(missing_ok=True)
                else:
                    rollback = path.with_name(
                        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.rollback"
                    )
                    rollback.write_bytes(previous)
                    os.replace(rollback, path)
                fsync_directory(path.parent)
            except OSError:
                pass
        raise
    finally:
        temporary.unlink(missing_ok=True)


def parse_byte_size(value: str | int) -> int:
    if isinstance(value, int):
        if value <= 0:
            raise argparse.ArgumentTypeError("byte size must be positive")
        return value
    match = _BYTE_SIZE_RE.fullmatch(value.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            f"invalid byte size {value!r}; use bytes or a k/m/g/t suffix"
        )
    number = Decimal(match.group(1))
    exponent = {"": 0, "k": 1, "m": 2, "g": 3, "t": 4}[match.group(2).lower()]
    byte_count = number * (1024**exponent)
    if byte_count != byte_count.to_integral_value() or byte_count <= 0:
        raise argparse.ArgumentTypeError("byte size must be a positive whole byte count")
    return int(byte_count)


def parse_cpus(value: str) -> tuple[str, int]:
    try:
        cpus = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid CPU count {value!r}") from exc
    nano_cpus = cpus * 1_000_000_000
    if cpus <= 0 or nano_cpus != nano_cpus.to_integral_value():
        raise argparse.ArgumentTypeError(
            "CPU count must be positive and expressible in whole NanoCPUs"
        )
    normalized = format(cpus.normalize(), "f")
    return normalized, int(nano_cpus)


def _port(value: str | int) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _positive_float(value: str | float) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _readonly_percentage(value: str | int) -> int:
    result = int(value)
    if not 1 <= result <= 100:
        raise argparse.ArgumentTypeError("read-only percentage must be in [1, 100]")
    return result


def config_from_start_args(args: argparse.Namespace) -> ServiceConfig:
    cpus, nano_cpus = parse_cpus(str(args.cpus))
    data_dir = args.data_dir.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    if not _NAME_RE.fullmatch(args.container_name):
        raise ServiceError(f"invalid container name: {args.container_name!r}")
    if not _NAME_RE.fullmatch(args.cluster_hostname):
        raise ServiceError(f"invalid cluster hostname: {args.cluster_hostname!r}")
    try:
        ipaddress.ip_address(args.bind_address)
    except ValueError as exc:
        raise ServiceError("--bind-address must be an IP address") from exc
    if args.rest_port == args.grpc_port:
        raise ServiceError("REST and gRPC host ports must be different")
    return ServiceConfig(
        container_name=args.container_name,
        cluster_hostname=args.cluster_hostname,
        data_dir=data_dir,
        manifest=manifest,
        bind_address=args.bind_address,
        rest_port=args.rest_port,
        grpc_port=args.grpc_port,
        cpus=cpus,
        nano_cpus=nano_cpus,
        memory_bytes=args.memory,
        disk_use_readonly_percentage=args.disk_use_readonly_percentage,
        min_free_bytes=args.min_free_bytes,
        startup_timeout=args.startup_timeout,
        poll_interval=args.poll_interval,
        http_timeout=args.http_timeout,
    )


def expected_environment(config: ServiceConfig) -> dict[str, str]:
    return {
        "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
        "CLUSTER_HOSTNAME": config.cluster_hostname,
        "DEFAULT_VECTORIZER_MODULE": "none",
        "DISK_USE_READONLY_PERCENTAGE": str(
            config.disk_use_readonly_percentage
        ),
        "GRPC_PORT": str(CONTAINER_GRPC_PORT),
        "PERSISTENCE_DATA_PATH": DATA_MOUNT_TARGET,
    }


def docker_publish_host(bind_address: str) -> str:
    return f"[{bind_address}]" if ":" in bind_address else bind_address


def build_docker_run_command(config: ServiceConfig) -> list[str]:
    command = [
        "docker",
        "run",
        "--detach",
        "--pull",
        "always",
        "--platform",
        DOCKER_PLATFORM,
        "--name",
        config.container_name,
        "--label",
        f"{SERVICE_LABEL_KEY}={SERVICE_LABEL_VALUE}",
        "--restart",
        "no",
        "--publish",
        f"{docker_publish_host(config.bind_address)}:{config.rest_port}:{CONTAINER_REST_PORT}/tcp",
        "--publish",
        f"{docker_publish_host(config.bind_address)}:{config.grpc_port}:{CONTAINER_GRPC_PORT}/tcp",
        "--mount",
        f"type=bind,source={config.data_dir},target={DATA_MOUNT_TARGET}",
        "--cpus",
        config.cpus,
        "--memory",
        str(config.memory_bytes),
    ]
    for key, value in expected_environment(config).items():
        command.extend(("--env", f"{key}={value}"))
    command.append(IMAGE_REFERENCE)
    return command


def dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    config = config_from_start_args(args)
    command = build_docker_run_command(config)
    return {
        "dry_run": True,
        "side_effects": {
            "filesystem_read": False,
            "filesystem_write": False,
            "docker_invoked": False,
            "network": False,
        },
        "service_configuration": config.manifest_configuration(),
        "docker_run_command": command,
        "docker_run_command_shell": shlex.join(command),
        "runtime_gates_when_executed": [
            "absolute and percentage disk headroom",
            "pinned image digest and linux/amd64 inspection",
            "container environment, mount, ports, CPU, and memory inspection",
            "Weaviate 1.38.0 REST metadata and gRPC listener readiness",
        ],
    }


def run_process(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise ServiceError(f"command failed ({' '.join(command)}): {detail}")
    return result


def docker_runtime_provenance() -> dict[str, Any]:
    result = run_process(("docker", "version", "--format", "{{json .}}"))
    try:
        docker_version = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ServiceError("docker version returned invalid JSON") from exc
    if not isinstance(docker_version, Mapping):
        raise ServiceError("docker version returned a non-object payload")
    return {
        "docker_version": dict(docker_version),
        "host_system": os.uname().sysname,
        "host_release": os.uname().release,
        "host_machine": os.uname().machine,
        "python_platform": sys.platform,
        "required_platform": DOCKER_PLATFORM,
    }


def _single_inspect_payload(result: subprocess.CompletedProcess[str], subject: str) -> dict[str, Any]:
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ServiceError(f"docker inspect returned invalid JSON for {subject}") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ServiceError(f"docker inspect returned an unexpected payload for {subject}")
    return payload[0]


def inspect_container(container_name: str) -> dict[str, Any] | None:
    result = run_process(
        ("docker", "inspect", "--type", "container", container_name), check=False
    )
    if result.returncode != 0:
        diagnostic = f"{result.stderr}\n{result.stdout}".lower()
        if "no such object" in diagnostic or "no such container" in diagnostic:
            return None
        raise ServiceError(
            f"cannot inspect container {container_name!r}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return _single_inspect_payload(result, container_name)


def inspect_image(image_id: str) -> dict[str, Any]:
    result = run_process(("docker", "image", "inspect", image_id))
    return _single_inspect_payload(result, image_id)


def filesystem_preflight(config: ServiceConfig) -> dict[str, Any]:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    if not config.data_dir.is_dir():
        raise ServiceError(f"data path is not a directory: {config.data_dir}")
    usage = shutil.disk_usage(config.data_dir)
    used_percentage = 100.0 * usage.used / usage.total
    if usage.free < config.min_free_bytes:
        raise ServiceError(
            "insufficient free space for formal Weaviate data: "
            f"required={config.min_free_bytes} actual={usage.free} bytes"
        )
    if used_percentage >= config.disk_use_readonly_percentage:
        raise ServiceError(
            "data filesystem is already at or above Weaviate's configured "
            f"read-only threshold: used={used_percentage:.3f}% "
            f"threshold={config.disk_use_readonly_percentage}%"
        )
    rationale = (
        f"The filesystem containing the formal data directory is {used_percentage:.3f}% "
        f"used. This host can exceed Weaviate's default {WEAVIATE_DEFAULT_DISK_USE_READONLY_PERCENTAGE}% "
        f"read-only threshold, so the explicit {config.disk_use_readonly_percentage}% threshold keeps "
        "the service writable for the formal corpus. The independent absolute free-space gate "
        f"({config.min_free_bytes} bytes minimum) prevents that percentage override from hiding "
        "insufficient storage headroom."
    )
    return {
        "path": str(config.data_dir),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_percentage": round(used_percentage, 6),
        "weaviate_default_readonly_percentage": (
            WEAVIATE_DEFAULT_DISK_USE_READONLY_PERCENTAGE
        ),
        "configured_readonly_percentage": config.disk_use_readonly_percentage,
        "minimum_free_bytes": config.min_free_bytes,
        "gates_passed": True,
        "rationale": rationale,
    }


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ServiceError(f"docker inspect field {field} is not an object")
    return value


def _environment_map(values: Any) -> dict[str, str]:
    if not isinstance(values, list):
        raise ServiceError("docker inspect Config.Env is not a list")
    environment: dict[str, str] = {}
    for item in values:
        if not isinstance(item, str) or "=" not in item:
            raise ServiceError("docker inspect contains a malformed environment entry")
        key, value = item.split("=", 1)
        if key in environment:
            raise ServiceError(f"docker inspect contains duplicate environment key {key!r}")
        environment[key] = value
    return environment


def verify_runtime_inspection(
    container: Mapping[str, Any], image: Mapping[str, Any], config: ServiceConfig
) -> dict[str, Any]:
    errors: list[str] = []
    container_config = _mapping(container.get("Config"), "Config")
    host_config = _mapping(container.get("HostConfig"), "HostConfig")
    state = _mapping(container.get("State"), "State")

    configured_image = container_config.get("Image")
    if configured_image != IMAGE_REFERENCE:
        errors.append(
            f"Config.Image expected={IMAGE_REFERENCE!r} actual={configured_image!r}"
        )
    repo_digests = image.get("RepoDigests")
    if not isinstance(repo_digests, list) or IMAGE_REFERENCE not in repo_digests:
        errors.append(
            f"RepoDigests does not contain the pinned identity {IMAGE_REFERENCE!r}"
        )
    if container.get("Image") != image.get("Id"):
        errors.append("container image ID does not match inspected image ID")
    if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
        errors.append(
            "resolved image platform expected='linux/amd64' "
            f"actual={image.get('Os')!r}/{image.get('Architecture')!r}"
        )

    actual_environment = _environment_map(container_config.get("Env"))
    expected_env = expected_environment(config)
    for key, expected in expected_env.items():
        actual = actual_environment.get(key)
        if actual != expected:
            errors.append(f"environment {key} expected={expected!r} actual={actual!r}")

    labels = container_config.get("Labels")
    if not isinstance(labels, Mapping) or labels.get(SERVICE_LABEL_KEY) != SERVICE_LABEL_VALUE:
        errors.append("formal-service ownership label is missing or incorrect")

    mounts = container.get("Mounts")
    matching_mounts = []
    if isinstance(mounts, list):
        matching_mounts = [
            mount
            for mount in mounts
            if isinstance(mount, Mapping)
            and mount.get("Destination") == DATA_MOUNT_TARGET
        ]
    if len(matching_mounts) != 1:
        errors.append(f"expected exactly one mount at {DATA_MOUNT_TARGET}")
        actual_mount: Mapping[str, Any] = {}
    else:
        actual_mount = matching_mounts[0]
        if actual_mount.get("Type") != "bind":
            errors.append("formal data mount is not a bind mount")
        try:
            source = Path(str(actual_mount.get("Source"))).resolve()
        except OSError:
            source = Path("/")
        if source != config.data_dir:
            errors.append(
                f"data mount source expected={config.data_dir} actual={source}"
            )
        if actual_mount.get("RW") is not True:
            errors.append("formal data mount is not writable")

    actual_nano_cpus = host_config.get("NanoCpus")
    actual_memory = host_config.get("Memory")
    if actual_nano_cpus != config.nano_cpus:
        errors.append(
            f"NanoCpus expected={config.nano_cpus} actual={actual_nano_cpus}"
        )
    if actual_memory != config.memory_bytes:
        errors.append(f"Memory expected={config.memory_bytes} actual={actual_memory}")

    port_bindings = host_config.get("PortBindings")
    if not isinstance(port_bindings, Mapping):
        errors.append("HostConfig.PortBindings is not an object")
        port_bindings = {}
    expected_ports = {
        f"{CONTAINER_REST_PORT}/tcp": config.rest_port,
        f"{CONTAINER_GRPC_PORT}/tcp": config.grpc_port,
    }
    for container_port, host_port in expected_ports.items():
        expected_binding = [
            {"HostIp": config.bind_address, "HostPort": str(host_port)}
        ]
        if port_bindings.get(container_port) != expected_binding:
            errors.append(
                f"port binding {container_port} expected={expected_binding!r} "
                f"actual={port_bindings.get(container_port)!r}"
            )

    restart_policy = host_config.get("RestartPolicy")
    if not isinstance(restart_policy, Mapping) or restart_policy.get("Name") != "no":
        errors.append("restart policy must be explicitly 'no'")
    if state.get("Running") is not True:
        errors.append("container is not running")
    if errors:
        raise ServiceError("docker runtime verification failed: " + "; ".join(errors))

    return {
        "validated": True,
        "container_id": container.get("Id"),
        "container_image_id": container.get("Image"),
        "configured_image": configured_image,
        "repo_digests": list(repo_digests),
        "resolved_platform": f"{image.get('Os')}/{image.get('Architecture')}",
        "environment": {key: actual_environment[key] for key in expected_env},
        "mount": dict(actual_mount),
        "resources": {
            "nano_cpus": actual_nano_cpus,
            "memory_bytes": actual_memory,
        },
        "port_bindings": dict(port_bindings),
        "restart_policy": dict(restart_policy),
    }


def _probe_host(bind_address: str) -> str:
    if bind_address == "0.0.0.0":
        return "127.0.0.1"
    if bind_address == "::":
        return "[::1]"
    if ":" in bind_address:
        return f"[{bind_address}]"
    return bind_address


def fetch_meta(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ServiceError("/v1/meta returned a non-object JSON payload")
    return payload


def probe_tcp_listener(host: str, port: int, timeout: float) -> None:
    with socket.create_connection((host, port), timeout=timeout):
        return


def wait_for_grpc_listener(
    config: ServiceConfig,
    *,
    connector: Callable[[str, int, float], None] = probe_tcp_listener,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    host = (
        "127.0.0.1"
        if config.bind_address == "0.0.0.0"
        else "::1" if config.bind_address == "::" else config.bind_address
    )
    deadline = monotonic() + config.startup_timeout
    attempts = 0
    last_error = "listener did not accept a connection"
    while True:
        attempts += 1
        try:
            connector(host, config.grpc_port, config.http_timeout)
        except (OSError, TimeoutError) as exc:
            last_error = str(exc)
        else:
            return {
                "host": host,
                "port": config.grpc_port,
                "attempts": attempts,
                "probe_scope": "tcp_listener_readiness_not_rpc_semantics",
                "listener_ready": True,
            }
        now = monotonic()
        if now >= deadline:
            raise ServiceError(
                "timed out waiting for Weaviate gRPC TCP listener at "
                f"{host}:{config.grpc_port} after {attempts} attempts: {last_error}"
            )
        sleeper(min(config.poll_interval, deadline - now))


def wait_for_meta(
    config: ServiceConfig,
    *,
    fetcher: Callable[[str, float], dict[str, Any]] = fetch_meta,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    url = f"http://{_probe_host(config.bind_address)}:{config.rest_port}/v1/meta"
    deadline = monotonic() + config.startup_timeout
    attempts = 0
    last_error = "service did not answer"
    while True:
        attempts += 1
        try:
            meta = fetcher(url, config.http_timeout)
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        else:
            actual_version = str(meta.get("version", ""))
            if actual_version != WEAVIATE_VERSION:
                raise ServiceError(
                    f"Weaviate version mismatch: expected={WEAVIATE_VERSION!r} "
                    f"actual={actual_version!r}"
                )
            return {
                "url": url,
                "attempts": attempts,
                "expected_version": WEAVIATE_VERSION,
                "actual_version": actual_version,
                "payload": meta,
            }
        now = monotonic()
        if now >= deadline:
            raise ServiceError(
                f"timed out waiting for {url} after {attempts} attempts: {last_error}"
            )
        sleeper(min(config.poll_interval, deadline - now))


def _container_is_owned(container: Mapping[str, Any]) -> bool:
    config = container.get("Config")
    if not isinstance(config, Mapping):
        return False
    labels = config.get("Labels")
    return isinstance(labels, Mapping) and labels.get(SERVICE_LABEL_KEY) == SERVICE_LABEL_VALUE


def _remove_container(container_name: str, *, stop_timeout: int = 30) -> None:
    container = inspect_container(container_name)
    if container is None:
        return
    if not _container_is_owned(container):
        raise ServiceError(
            f"refusing to remove unowned container named {container_name!r}"
        )
    state = container.get("State")
    if isinstance(state, Mapping) and state.get("Running") is True:
        run_process(("docker", "stop", "--time", str(stop_timeout), container_name))
    # No --volumes flag: the formal bind-mounted data directory is retained.
    run_process(("docker", "rm", container_name))


def validate_runtime_manifest(
    payload: Mapping[str, Any],
    config: ServiceConfig,
    inspection: Mapping[str, Any] | None = None,
) -> None:
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ServiceError("runtime manifest schema is absent or incompatible")
    if payload.get("artifact") != "weaviate_amazon10m_formal_service_runtime":
        raise ServiceError("runtime manifest artifact identity is invalid")
    if payload.get("status") != "running" or payload.get("identity_valid") is not True:
        raise ServiceError("runtime manifest does not describe a verified running service")
    if payload.get("configuration") != config.manifest_configuration():
        raise ServiceError("runtime manifest configuration differs from the requested fixed contract")
    recorded_runtime = payload.get("runtime_provenance")
    if not isinstance(recorded_runtime, Mapping) or recorded_runtime.get(
        "required_platform"
    ) != DOCKER_PLATFORM:
        raise ServiceError("runtime manifest is missing Docker/host platform provenance")
    if inspection is not None:
        recorded_inspection = payload.get("docker_inspection")
        if not isinstance(recorded_inspection, Mapping):
            raise ServiceError("runtime manifest is missing Docker inspection evidence")
        if recorded_inspection.get("container_id") != inspection.get("container_id"):
            raise ServiceError("runtime manifest belongs to a different container instance")


def build_running_manifest(
    config: ServiceConfig,
    disk: Mapping[str, Any],
    command: Sequence[str],
    reported_container_id: str,
    inspection: Mapping[str, Any],
    meta: Mapping[str, Any],
    grpc: Mapping[str, Any],
    runtime_provenance: Mapping[str, Any],
    start_action: str,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact": "weaviate_amazon10m_formal_service_runtime",
        "status": "running",
        "recorded_at": utc_now(),
        "start_action": start_action,
        "configuration": config.manifest_configuration(),
        "disk_preflight": dict(disk),
        "docker_run_command": list(command),
        "docker_reported_container_id": reported_container_id,
        "docker_inspection": dict(inspection),
        "weaviate_meta": dict(meta),
        "grpc_listener": dict(grpc),
        "runtime_provenance": dict(runtime_provenance),
        "identity_valid": True,
        "data_retained_on_stop": True,
    }


def start_service(args: argparse.Namespace) -> dict[str, Any]:
    config = config_from_start_args(args)
    disk = filesystem_preflight(config)
    existing = inspect_container(config.container_name)
    if existing is not None:
        if not _container_is_owned(existing):
            raise ServiceError(
                f"container {config.container_name!r} exists but is not owned by this controller"
            )
        image = inspect_image(str(existing.get("Image", "")))
        inspection = verify_runtime_inspection(existing, image, config)
        meta = wait_for_meta(config)
        grpc = wait_for_grpc_listener(config)
        runtime = docker_runtime_provenance()
        if config.manifest.exists():
            validate_runtime_manifest(
                _read_manifest(config.manifest), config, inspection
            )
        manifest = build_running_manifest(
            config,
            disk,
            build_docker_run_command(config),
            str(inspection.get("container_id", "")),
            inspection,
            meta,
            grpc,
            runtime,
            "reused_existing_verified_container",
        )
        atomic_write_json(config.manifest, manifest)
        return manifest

    command = build_docker_run_command(config)
    attempted_start = False
    try:
        attempted_start = True
        result = run_process(command)
        reported_container_id = result.stdout.strip()
        container = inspect_container(config.container_name)
        if container is None:
            raise ServiceError("docker run returned but the container cannot be inspected")
        image = inspect_image(str(container.get("Image", "")))
        inspection = verify_runtime_inspection(container, image, config)
        meta = wait_for_meta(config)
        grpc = wait_for_grpc_listener(config)
        runtime = docker_runtime_provenance()
        manifest = build_running_manifest(
            config,
            disk,
            command,
            reported_container_id,
            inspection,
            meta,
            grpc,
            runtime,
            "created_new_container",
        )
        atomic_write_json(config.manifest, manifest)
        return manifest
    except Exception:
        if attempted_start:
            try:
                _remove_container(config.container_name)
            except ServiceError:
                pass
            try:
                config.manifest.unlink(missing_ok=True)
                fsync_directory(config.manifest.parent)
            except OSError:
                pass
        raise


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ServiceError(f"runtime manifest does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ServiceError(f"runtime manifest is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ServiceError("runtime manifest must contain a JSON object")
    return payload


def _config_from_manifest(
    payload: Mapping[str, Any], manifest_path: Path, http_timeout: float
) -> ServiceConfig:
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ServiceError("runtime manifest is missing configuration")
    if configuration.get("image_reference") != IMAGE_REFERENCE:
        raise ServiceError("runtime manifest does not contain the fixed image identity")
    if configuration.get("weaviate_version") != WEAVIATE_VERSION:
        raise ServiceError("runtime manifest contains the wrong Weaviate version")
    try:
        return ServiceConfig(
            container_name=str(configuration["container_name"]),
            cluster_hostname=str(configuration["cluster_hostname"]),
            data_dir=Path(str(configuration["data_directory"])).resolve(),
            manifest=manifest_path,
            bind_address=str(configuration["bind_address"]),
            rest_port=int(configuration["rest_host_port"]),
            grpc_port=int(configuration["grpc_host_port"]),
            cpus=str(configuration["cpus"]),
            nano_cpus=int(configuration["nano_cpus"]),
            memory_bytes=int(configuration["memory_bytes"]),
            disk_use_readonly_percentage=int(
                configuration["disk_use_readonly_percentage"]
            ),
            min_free_bytes=int(configuration["minimum_free_bytes_before_start"]),
            startup_timeout=http_timeout,
            poll_interval=http_timeout,
            http_timeout=http_timeout,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ServiceError("runtime manifest configuration is malformed") from exc


def status_service(args: argparse.Namespace) -> dict[str, Any]:
    config = config_from_start_args(args)
    manifest_path = config.manifest
    container = inspect_container(config.container_name)
    if container is None:
        return {
            "status": "stopped",
            "container_name": config.container_name,
            "container_present": False,
            "manifest": str(manifest_path),
        }
    payload = _read_manifest(manifest_path)
    image = inspect_image(str(container.get("Image", "")))
    inspection = verify_runtime_inspection(container, image, config)
    validate_runtime_manifest(payload, config, inspection)
    meta = fetch_meta(
        f"http://{_probe_host(config.bind_address)}:{config.rest_port}/v1/meta",
        config.http_timeout,
    )
    if str(meta.get("version", "")) != WEAVIATE_VERSION:
        raise ServiceError(
            f"Weaviate version mismatch: expected={WEAVIATE_VERSION!r} "
            f"actual={meta.get('version')!r}"
        )
    grpc = wait_for_grpc_listener(config)
    runtime = docker_runtime_provenance()
    return {
        "status": "running",
        "container_name": config.container_name,
        "container_id": inspection["container_id"],
        "image_reference": IMAGE_REFERENCE,
        "weaviate_version": WEAVIATE_VERSION,
        "identity_valid": True,
        "grpc_listener": grpc,
        "runtime_provenance": runtime,
        "manifest": str(manifest_path),
    }


def stop_service(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve()
    container = inspect_container(args.container_name)
    container_present = container is not None
    if container is not None:
        if not _container_is_owned(container):
            raise ServiceError(
                f"refusing to stop unowned container named {args.container_name!r}"
            )
        _remove_container(args.container_name, stop_timeout=args.stop_timeout)

    if manifest_path.exists():
        manifest = _read_manifest(manifest_path)
    else:
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "artifact": "weaviate_amazon10m_formal_service_runtime",
            "configuration": {
                "container_name": args.container_name,
                "image_reference": IMAGE_REFERENCE,
                "image_digest": IMAGE_DIGEST,
                "weaviate_version": WEAVIATE_VERSION,
                "data_directory": str(DEFAULT_DATA_DIR.resolve()),
            },
        }
    manifest.update(
        {
            "status": "stopped",
            "stopped_at": utc_now(),
            "container_removed": container_present,
            "data_retained": True,
        }
    )
    atomic_write_json(manifest_path, manifest)
    return {
        "status": "stopped",
        "container_name": args.container_name,
        "container_removed": container_present,
        "data_retained": True,
        "manifest": str(manifest_path),
    }


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--container-name", default=CONTAINER_NAME)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cluster-hostname", default=CLUSTER_HOSTNAME)
    parser.add_argument("--bind-address", default=DEFAULT_BIND_ADDRESS)
    parser.add_argument("--rest-port", type=_port, default=DEFAULT_REST_PORT)
    parser.add_argument("--grpc-port", type=_port, default=DEFAULT_GRPC_PORT)
    parser.add_argument("--cpus", default=DEFAULT_CPUS)
    parser.add_argument("--memory", type=parse_byte_size, default=parse_byte_size(DEFAULT_MEMORY))
    parser.add_argument(
        "--disk-use-readonly-percentage",
        type=_readonly_percentage,
        default=DEFAULT_DISK_USE_READONLY_PERCENTAGE,
    )
    parser.add_argument(
        "--min-free-bytes",
        type=parse_byte_size,
        default=DEFAULT_MIN_FREE_BYTES,
    )
    parser.add_argument("--startup-timeout", type=_positive_float, default=180.0)
    parser.add_argument("--poll-interval", type=_positive_float, default=1.0)
    parser.add_argument("--http-timeout", type=_positive_float, default=5.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control the pinned Weaviate 1.38.0 Amazon-10M formal service"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="start and verify the formal service")
    _add_identity_arguments(start)
    _add_runtime_arguments(start)
    start.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact Docker command and gates without side effects",
    )
    start.set_defaults(handler=start_service)

    status = subparsers.add_parser("status", help="inspect and verify the running service")
    _add_identity_arguments(status)
    _add_runtime_arguments(status)
    status.set_defaults(handler=status_service)

    stop = subparsers.add_parser("stop", help="stop the container and retain formal data")
    _add_identity_arguments(stop)
    stop.add_argument("--stop-timeout", type=int, default=30)
    stop.set_defaults(handler=stop_service)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "dry_run", False):
        print(json.dumps(dry_run_payload(args), sort_keys=True))
        return 0
    try:
        result = args.handler(args)
    except (ServiceError, OSError, ValueError) as exc:
        print(f"formal Weaviate service failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
