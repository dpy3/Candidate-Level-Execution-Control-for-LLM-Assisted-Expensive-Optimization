from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .aggregate_uav_results import export_uav_suite_results
    from .analyze_uav_results import analyze_uav_suite
    from .build_uav_suite import build_uav_suite_manifest
    from .run_uav_benchmark import run_uav_benchmark

__all__ = [
    "build_uav_suite_manifest",
    "run_uav_benchmark",
    "export_uav_suite_results",
    "analyze_uav_suite",
]


def __getattr__(name: str):
    if name == "build_uav_suite_manifest":
        from .build_uav_suite import build_uav_suite_manifest as value

        return value
    if name == "run_uav_benchmark":
        from .run_uav_benchmark import run_uav_benchmark as value

        return value
    if name == "export_uav_suite_results":
        from .aggregate_uav_results import export_uav_suite_results as value

        return value
    if name == "analyze_uav_suite":
        from .analyze_uav_results import analyze_uav_suite as value

        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
