import os
from pathlib import Path

from safe_llm_saea.package_results import package_all
from safe_llm_saea.paper_ready_boxplot_visualizer import plot_box
from safe_llm_saea.paper_ready_convergence_visualizer import plot_convergence
from safe_llm_saea.paper_ready_performance_profile import plot_performance_profile


def main():
    base_dir = Path(__file__).resolve().parents[1]
    json_path = str(base_dir / "safe_llm_saea" / "lightweight_paper_results" / "experiment_results_latest.json")
    out_dir = str(base_dir / "safe_llm_saea" / "lightweight_paper_results" / "analysis_final")

    package_all(json_path, out_dir)
    plot_box(json_path, out_dir)
    plot_convergence(json_path, out_dir)
    plot_performance_profile(json_path, out_dir)


if __name__ == "__main__":
    main()
