from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .aggregate_wsn_results import aggregate_wsn_pipeline, export_wsn_suite_results
    from .analyze_wsn_results import analyze_wsn_suite
    from .build_wsn_suite import build_wsn_suite_manifest
    from .run_wsn_benchmark import run_wsn_benchmark

__all__ = [
    "build_wsn_suite_manifest",
    "run_wsn_benchmark",
    "aggregate_wsn_pipeline",
    "export_wsn_suite_results",
    "analyze_wsn_suite",
]


def __getattr__(name: str):
    if name == "build_wsn_suite_manifest":
        from .build_wsn_suite import build_wsn_suite_manifest as value

        return value
    if name == "run_wsn_benchmark":
        from .run_wsn_benchmark import run_wsn_benchmark as value

        return value
    if name == "aggregate_wsn_pipeline":
        from .aggregate_wsn_results import aggregate_wsn_pipeline as value

        return value
    if name == "export_wsn_suite_results":
        from .aggregate_wsn_results import export_wsn_suite_results as value

        return value
    if name == "analyze_wsn_suite":
        from .analyze_wsn_results import analyze_wsn_suite as value

        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
