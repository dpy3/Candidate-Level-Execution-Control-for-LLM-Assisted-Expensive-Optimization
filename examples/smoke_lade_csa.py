from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safe_llm_saea.paper_ready_safe_llm_saea import ExperimentConfig, LLMAssistedDE


def sphere(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))


def main():
    dim = 5
    config = ExperimentConfig(
        algorithm_name="L-ADE-CSA",
        problem_name="Sphere",
        dimension=dim,
        max_evaluations=120,
        population_size=12,
        llm_enhancement=False,
        gating_enabled=True,
        max_llm_calls_per_run=0,
    )
    optimizer = LLMAssistedDE(config=config, use_api=False)
    result = optimizer.optimize(sphere, [(-5.0, 5.0)] * dim)
    print("best_fitness:", result["best_fitness"])
    print("evaluations_used:", result["evaluations_used"])


if __name__ == "__main__":
    main()
