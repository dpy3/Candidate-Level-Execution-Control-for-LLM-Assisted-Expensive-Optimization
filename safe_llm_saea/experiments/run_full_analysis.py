import os
import sys
from pathlib import Path

# Add project root to path to allow importing safe_llm_saea
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from safe_llm_saea.package_results import package_all
from safe_llm_saea.paper_ready_boxplot_visualizer import plot_box
from safe_llm_saea.paper_ready_convergence_visualizer import plot_convergence
from safe_llm_saea.paper_ready_performance_profile import plot_performance_profile


def main():
    base_dir = Path(__file__).resolve().parents[1]
    json_path = str(
        base_dir / "lightweight_paper_results" / "experiment_results_latest.json"
    )
    
    if not os.path.exists(json_path):
        # Fallback to verify other locations if moved
        print(f"Warning: {json_path} not found. Checking alternatives...")
        project_root = Path(__file__).resolve().parents[2]
        alt_path = project_root / "safe_llm_saea" / "lightweight_paper_results" / "experiment_results_latest.json"
        if os.path.exists(alt_path):
            json_path = str(alt_path)
        else:
             raise FileNotFoundError(f"experiment_results_latest.json not found at {json_path} or {alt_path}")

    print(f"Using results from: {json_path}")
    out_dir = os.path.join(os.path.dirname(json_path), "analysis_final")

    package_all(json_path, out_dir)
    plot_box(json_path, out_dir)
    plot_convergence(json_path, out_dir)
    plot_performance_profile(json_path, out_dir)


if __name__ == "__main__":
    main()
