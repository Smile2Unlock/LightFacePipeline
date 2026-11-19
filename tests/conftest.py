"""Pytest session configuration: clamp OpenMP threads inside container cgroups."""

from __future__ import annotations

from common.runtime import configure_cpu_threads


def pytest_configure(config) -> None:  # noqa: ARG001
    configure_cpu_threads(4)
