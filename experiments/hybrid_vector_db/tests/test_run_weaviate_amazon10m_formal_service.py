from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "experiments/hybrid_vector_db/scripts/run_weaviate_amazon10m_formal_service.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_weaviate_amazon10m_formal_service", SCRIPT
)
service = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = service
SPEC.loader.exec_module(service)


def make_config(directory: Path, **overrides) -> service.ServiceConfig:
    values = {
        "container_name": service.CONTAINER_NAME,
        "cluster_hostname": service.CLUSTER_HOSTNAME,
        "data_dir": (directory / "data").resolve(),
        "manifest": (directory / "runtime.json").resolve(),
        "bind_address": "127.0.0.1",
        "rest_port": 18080,
        "grpc_port": 15051,
        "cpus": "2.5",
        "nano_cpus": 2_500_000_000,
        "memory_bytes": 3 * 1024**3,
        "disk_use_readonly_percentage": 98,
        "min_free_bytes": 1024,
        "startup_timeout": 2.0,
        "poll_interval": 0.01,
        "http_timeout": 0.5,
    }
    values.update(overrides)
    return service.ServiceConfig(**values)


def make_inspections(config: service.ServiceConfig):
    image_id = "sha256:" + "b" * 64
    environment = [
        f"{key}={value}"
        for key, value in service.expected_environment(config).items()
    ]
    container = {
        "Id": "container-id",
        "Image": image_id,
        "Config": {
            "Image": service.IMAGE_REFERENCE,
            "Env": environment,
            "Labels": {
                service.SERVICE_LABEL_KEY: service.SERVICE_LABEL_VALUE,
            },
        },
        "HostConfig": {
            "NanoCpus": config.nano_cpus,
            "Memory": config.memory_bytes,
            "PortBindings": {
                f"{service.CONTAINER_REST_PORT}/tcp": [
                    {
                        "HostIp": config.bind_address,
                        "HostPort": str(config.rest_port),
                    }
                ],
                f"{service.CONTAINER_GRPC_PORT}/tcp": [
                    {
                        "HostIp": config.bind_address,
                        "HostPort": str(config.grpc_port),
                    }
                ],
            },
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": str(config.data_dir),
                "Destination": service.DATA_MOUNT_TARGET,
                "Mode": "",
                "RW": True,
                "Propagation": "rprivate",
            }
        ],
        "State": {"Running": True},
    }
    image = {
        "Id": image_id,
        "RepoDigests": [service.IMAGE_REFERENCE],
        "Os": "linux",
        "Architecture": "amd64",
    }
    return container, image


def make_args(config: service.ServiceConfig) -> argparse.Namespace:
    return argparse.Namespace(
        container_name=config.container_name,
        cluster_hostname=config.cluster_hostname,
        data_dir=config.data_dir,
        manifest=config.manifest,
        bind_address=config.bind_address,
        rest_port=config.rest_port,
        grpc_port=config.grpc_port,
        cpus=config.cpus,
        memory=config.memory_bytes,
        disk_use_readonly_percentage=config.disk_use_readonly_percentage,
        min_free_bytes=config.min_free_bytes,
        startup_timeout=config.startup_timeout,
        poll_interval=config.poll_interval,
        http_timeout=config.http_timeout,
    )


class RunWeaviateAmazon10MFormalServiceTests(unittest.TestCase):
    def test_start_dry_run_prints_exact_command_without_side_effects(self) -> None:
        args = service.build_parser().parse_args([
            "start", "--dry-run", "--data-dir", "/missing/weaviate-data",
            "--manifest", "/missing/runtime.json",
        ])
        with (
            mock.patch.object(service, "run_process", side_effect=AssertionError("process")),
            mock.patch.object(service.shutil, "disk_usage", side_effect=AssertionError("disk")),
            mock.patch.object(Path, "open", side_effect=AssertionError("file")),
        ):
            payload = service.dry_run_payload(args)
        self.assertTrue(payload["dry_run"])
        self.assertFalse(payload["side_effects"]["docker_invoked"])
        self.assertEqual(payload["docker_run_command"][0:2], ["docker", "run"])
        self.assertEqual(payload["docker_run_command"][-1], service.IMAGE_REFERENCE)
        self.assertIn("--cpus", payload["docker_run_command"])

    def test_image_identity_is_fixed_digest_and_command_has_formal_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            command = service.build_docker_run_command(config)

        self.assertEqual(command[-1], service.IMAGE_REFERENCE)
        self.assertEqual(
            service.IMAGE_DIGEST,
            "sha256:5ec2f15768eb59d9f5ea21edb29b6395d9844d9641caa694e33e8689f65fee0f",
        )
        self.assertNotIn("latest", " ".join(command).lower())
        self.assertEqual(command[command.index("--platform") + 1], "linux/amd64")
        self.assertIn("127.0.0.1:18080:8080/tcp", command)
        self.assertIn("127.0.0.1:15051:50051/tcp", command)
        self.assertIn("AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true", command)
        self.assertIn("DEFAULT_VECTORIZER_MODULE=none", command)
        self.assertIn(f"CLUSTER_HOSTNAME={service.CLUSTER_HOSTNAME}", command)
        self.assertIn("DISK_USE_READONLY_PERCENTAGE=98", command)
        self.assertIn("2.5", command)
        self.assertIn(str(3 * 1024**3), command)

    def test_defaults_use_independent_data_directory_and_explicit_resources(self) -> None:
        args = service.build_parser().parse_args(["start"])
        config = service.config_from_start_args(args)
        self.assertEqual(
            config.data_dir,
            (ROOT / "data/weaviate-amazon10m-formal").resolve(),
        )
        self.assertEqual(config.rest_port, 8080)
        self.assertEqual(config.grpc_port, 50051)
        self.assertGreater(config.nano_cpus, 0)
        self.assertGreater(config.memory_bytes, 0)
        self.assertEqual(config.disk_use_readonly_percentage, 98)
        self.assertGreater(config.min_free_bytes, 0)

    def test_disk_preflight_records_rationale_and_enforces_absolute_free_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory), min_free_bytes=100)
            usage = shutil._ntuple_diskusage(total=1_000, used=800, free=200)
            with mock.patch.object(service.shutil, "disk_usage", return_value=usage):
                record = service.filesystem_preflight(config)
            self.assertEqual(record["configured_readonly_percentage"], 98)
            self.assertEqual(record["minimum_free_bytes"], 100)
            self.assertIn("default 90%", record["rationale"])
            self.assertIn("absolute free-space gate", record["rationale"])

            too_full = shutil._ntuple_diskusage(total=1_000, used=901, free=99)
            with mock.patch.object(service.shutil, "disk_usage", return_value=too_full):
                with self.assertRaisesRegex(service.ServiceError, "insufficient free space"):
                    service.filesystem_preflight(config)

            over_threshold = shutil._ntuple_diskusage(total=2_000, used=1_970, free=100)
            with mock.patch.object(
                service.shutil, "disk_usage", return_value=over_threshold
            ):
                with self.assertRaisesRegex(service.ServiceError, "read-only threshold"):
                    service.filesystem_preflight(config)

    def test_inspection_verifies_repo_digest_environment_mount_and_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            container, image = make_inspections(config)
            actual = service.verify_runtime_inspection(container, image, config)
            self.assertTrue(actual["validated"])
            self.assertEqual(actual["repo_digests"], [service.IMAGE_REFERENCE])
            self.assertEqual(actual["resources"]["nano_cpus"], config.nano_cpus)
            self.assertEqual(actual["mount"]["Source"], str(config.data_dir))

            bad_image = deepcopy(image)
            bad_image["RepoDigests"] = [
                service.IMAGE_REPOSITORY + "@sha256:" + "c" * 64
            ]
            with self.assertRaisesRegex(service.ServiceError, "RepoDigests"):
                service.verify_runtime_inspection(container, bad_image, config)

            bad_environment = deepcopy(container)
            bad_environment["Config"]["Env"] = [
                item
                for item in bad_environment["Config"]["Env"]
                if not item.startswith("DEFAULT_VECTORIZER_MODULE=")
            ]
            bad_environment["Config"]["Env"].append(
                "DEFAULT_VECTORIZER_MODULE=text2vec-openai"
            )
            with self.assertRaisesRegex(service.ServiceError, "DEFAULT_VECTORIZER_MODULE"):
                service.verify_runtime_inspection(bad_environment, image, config)

            bad_resources = deepcopy(container)
            bad_resources["HostConfig"]["Memory"] -= 1
            with self.assertRaisesRegex(service.ServiceError, "Memory"):
                service.verify_runtime_inspection(bad_resources, image, config)

    def test_meta_poll_requires_exact_1_38_0(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            fetcher = mock.Mock(side_effect=[OSError("not ready"), {"version": "1.38.0"}])
            sleeper = mock.Mock()
            clock = mock.Mock(side_effect=[0.0, 0.1])
            result = service.wait_for_meta(
                config, fetcher=fetcher, sleeper=sleeper, monotonic=clock
            )
            self.assertEqual(result["actual_version"], "1.38.0")
            self.assertEqual(result["attempts"], 2)
            sleeper.assert_called_once()

            with self.assertRaisesRegex(service.ServiceError, "version mismatch"):
                service.wait_for_meta(
                    config,
                    fetcher=lambda _url, _timeout: {"version": "1.37.0"},
                )

    def test_grpc_listener_poll_is_independent_from_rest_meta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            connector = mock.Mock(side_effect=[OSError("not ready"), None])
            sleeper = mock.Mock()
            clock = mock.Mock(side_effect=[0.0, 0.1])

            evidence = service.wait_for_grpc_listener(
                config,
                connector=connector,
                sleeper=sleeper,
                monotonic=clock,
            )

        self.assertTrue(evidence["listener_ready"])
        self.assertEqual(evidence["attempts"], 2)
        self.assertEqual(
            evidence["probe_scope"], "tcp_listener_readiness_not_rpc_semantics"
        )

    def test_ipv6_publish_addresses_are_bracketed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory), bind_address="::1")
            command = service.build_docker_run_command(config)
        self.assertIn("[::1]:18080:8080/tcp", command)
        self.assertIn("[::1]:15051:50051/tcp", command)

    def test_start_writes_atomic_runtime_manifest_after_both_identity_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            container, image = make_inspections(config)
            args = make_args(config)
            disk = {"gates_passed": True, "rationale": "tested"}
            completed = subprocess.CompletedProcess(
                args=["docker", "run"], returncode=0, stdout="container-id\n", stderr=""
            )
            with (
                mock.patch.object(service, "filesystem_preflight", return_value=disk),
                mock.patch.object(
                    service, "inspect_container", side_effect=[None, container]
                ),
                mock.patch.object(service, "inspect_image", return_value=image),
                mock.patch.object(
                    service,
                    "wait_for_meta",
                    return_value={
                        "actual_version": "1.38.0",
                        "expected_version": "1.38.0",
                    },
                ),
                mock.patch.object(
                    service,
                    "wait_for_grpc_listener",
                    return_value={"listener_ready": True},
                ),
                mock.patch.object(
                    service,
                    "docker_runtime_provenance",
                    return_value={"required_platform": service.DOCKER_PLATFORM},
                ),
                mock.patch.object(service, "run_process", return_value=completed) as runner,
            ):
                manifest = service.start_service(args)

            persisted = json.loads(config.manifest.read_text(encoding="utf-8"))
            self.assertEqual(persisted, manifest)
            self.assertEqual(persisted["status"], "running")
            self.assertTrue(persisted["identity_valid"])
            self.assertEqual(
                persisted["configuration"]["image_reference"],
                service.IMAGE_REFERENCE,
            )
            self.assertEqual(
                persisted["docker_inspection"]["environment"]
                ["DEFAULT_VECTORIZER_MODULE"],
                "none",
            )
            self.assertFalse(any(config.manifest.parent.glob("*.tmp")))
            self.assertEqual(runner.call_count, 1)

    def test_start_reuses_only_an_existing_fully_verified_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            args = make_args(config)
            container, image = make_inspections(config)
            inspection = service.verify_runtime_inspection(container, image, config)
            prior = service.build_running_manifest(
                config,
                {"gates_passed": True},
                service.build_docker_run_command(config),
                "container-id",
                inspection,
                {"actual_version": service.WEAVIATE_VERSION},
                {"listener_ready": True},
                {"required_platform": service.DOCKER_PLATFORM},
                "created_new_container",
            )
            service.atomic_write_json(config.manifest, prior)
            with (
                mock.patch.object(service, "filesystem_preflight", return_value={"gates_passed": True}),
                mock.patch.object(service, "inspect_container", return_value=container),
                mock.patch.object(service, "inspect_image", return_value=image),
                mock.patch.object(service, "wait_for_meta", return_value={"actual_version": service.WEAVIATE_VERSION}),
                mock.patch.object(service, "wait_for_grpc_listener", return_value={"listener_ready": True}),
                mock.patch.object(service, "docker_runtime_provenance", return_value={"required_platform": service.DOCKER_PLATFORM}),
                mock.patch.object(service, "run_process") as runner,
            ):
                manifest = service.start_service(args)

        self.assertEqual(manifest["start_action"], "reused_existing_verified_container")
        runner.assert_not_called()

    def test_atomic_manifest_restores_previous_state_when_directory_fsync_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text('{"status":"stopped"}\n', encoding="utf-8")
            with mock.patch.object(
                service,
                "fsync_directory",
                side_effect=[OSError("fsync failed"), None],
            ):
                with self.assertRaisesRegex(OSError, "fsync failed"):
                    service.atomic_write_json(path, {"status": "running"})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "stopped"},
            )

    def test_status_uses_manifest_to_reverify_live_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            container, image = make_inspections(config)
            inspection = service.verify_runtime_inspection(container, image, config)
            service.atomic_write_json(
                config.manifest,
                service.build_running_manifest(
                    config,
                    {"gates_passed": True},
                    service.build_docker_run_command(config),
                    "container-id",
                    inspection,
                    {"actual_version": "1.38.0"},
                    {"listener_ready": True},
                    {"required_platform": service.DOCKER_PLATFORM},
                    "created_new_container",
                ),
            )
            args = make_args(config)
            with (
                mock.patch.object(service, "inspect_container", return_value=container),
                mock.patch.object(service, "inspect_image", return_value=image),
                mock.patch.object(
                    service, "fetch_meta", return_value={"version": "1.38.0"}
                ),
                mock.patch.object(
                    service,
                    "wait_for_grpc_listener",
                    return_value={"listener_ready": True},
                ),
                mock.patch.object(
                    service,
                    "docker_runtime_provenance",
                    return_value={"required_platform": service.DOCKER_PLATFORM},
                ),
            ):
                result = service.status_service(args)
            self.assertEqual(result["status"], "running")
            self.assertTrue(result["identity_valid"])

    def test_status_rejects_mutated_manifest_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            container, image = make_inspections(config)
            inspection = service.verify_runtime_inspection(container, image, config)
            manifest = service.build_running_manifest(
                config,
                {"gates_passed": True},
                service.build_docker_run_command(config),
                "container-id",
                inspection,
                {"actual_version": service.WEAVIATE_VERSION},
                {"listener_ready": True},
                {"required_platform": service.DOCKER_PLATFORM},
                "created_new_container",
            )
            manifest["configuration"]["rest_host_port"] += 1
            service.atomic_write_json(config.manifest, manifest)
            with (
                mock.patch.object(service, "inspect_container", return_value=container),
                mock.patch.object(service, "inspect_image", return_value=image),
            ):
                with self.assertRaisesRegex(service.ServiceError, "configuration differs"):
                    service.status_service(make_args(config))

    def test_stop_removes_only_container_and_retains_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            config.data_dir.mkdir()
            marker = config.data_dir / "corpus.marker"
            marker.write_text("keep", encoding="utf-8")
            service.atomic_write_json(
                config.manifest,
                {
                    "schema_version": 1,
                    "status": "running",
                    "configuration": config.manifest_configuration(),
                },
            )
            container, _image = make_inspections(config)
            args = argparse.Namespace(
                container_name=config.container_name,
                manifest=config.manifest,
                stop_timeout=7,
            )
            calls = []

            def record(command, *, check=True):
                calls.append(list(command))
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                mock.patch.object(
                    service, "inspect_container", side_effect=[container, container]
                ),
                mock.patch.object(service, "run_process", side_effect=record),
            ):
                result = service.stop_service(args)

            self.assertEqual(
                calls,
                [
                    ["docker", "stop", "--time", "7", config.container_name],
                    ["docker", "rm", config.container_name],
                ],
            )
            self.assertTrue(marker.exists())
            self.assertTrue(result["data_retained"])
            persisted = json.loads(config.manifest.read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "stopped")
            self.assertTrue(persisted["data_retained"])


if __name__ == "__main__":
    unittest.main()
