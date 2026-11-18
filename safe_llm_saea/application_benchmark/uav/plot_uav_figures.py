from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from safe_llm_saea.application_benchmark.common.io_utils import ensure_dir, load_json


def _ordered_algorithms(analysis: Dict[str, Any]) -> List[str]:
    algorithms = analysis.get("algorithms") or list(analysis.get("success_rate", {}).keys())
    return [str(name) for name in algorithms]


def plot_uav_success_rates(analysis_path: str, output_dir: str) -> None:
    analysis = load_json(analysis_path)
    success_rate = analysis.get("success_rate", {})
    algorithms = _ordered_algorithms(analysis)
    values = [success_rate[name] for name in algorithms]
    ensure_dir(output_dir)
    plt.figure(figsize=(10, 5))
    plt.bar(algorithms, values)
    plt.ylim(0, 1.05)
    plt.ylabel("Strict success rate")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "uav_success_rate.png", dpi=300)
    plt.close()


def plot_uav_average_ranks(analysis_path: str, output_dir: str) -> None:
    analysis = load_json(analysis_path)
    average_rank = analysis.get("rank_table", {}).get("average_rank", {})
    algorithms = _ordered_algorithms(analysis)
    values = [average_rank[name] for name in algorithms]
    ensure_dir(output_dir)
    plt.figure(figsize=(10, 5))
    plt.bar(algorithms, values)
    plt.ylabel("Average rank")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "uav_average_rank.png", dpi=300)
    plt.close()


def plot_uav_feasibility_collision(analysis_path: str, output_dir: str) -> None:
    analysis = load_json(analysis_path)
    algorithms = _ordered_algorithms(analysis)
    success_rate = analysis.get("success_rate", {})
    collision_rate = analysis.get("collision_rate", {})
    collision_free_values = [float(collision_rate.get(name, {}).get("collision_free_rate", 0.0)) for name in algorithms]
    avg_collision_values = [float(collision_rate.get(name, {}).get("avg_collision_count", 0.0)) for name in algorithms]
    success_values = [float(success_rate.get(name, 0.0)) for name in algorithms]
    indices = np.arange(len(algorithms))

    ensure_dir(output_dir)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(indices - 0.18, success_values, width=0.36, label="Strict feasible rate", color="#4c78a8")
    axes[0].bar(indices + 0.18, collision_free_values, width=0.36, label="Collision-free rate", color="#72b7b2")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels(algorithms, rotation=25, ha="right")
    axes[0].set_ylabel("Rate")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(algorithms, avg_collision_values, color="#e45756")
    axes[1].set_ylabel("Average collision count")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.2)

    figure.tight_layout()
    figure.savefig(Path(output_dir) / "uav_feasibility_collision.png", dpi=300)
    plt.close(figure)


def plot_uav_case_cost_heatmap(analysis_path: str, output_dir: str) -> None:
    analysis = load_json(analysis_path)
    algorithms = _ordered_algorithms(analysis)
    per_case_mean_cost = analysis.get("per_case_mean_cost", {})
    case_ids = list(per_case_mean_cost.keys())
    if not algorithms or not case_ids:
        return
    matrix = np.asarray(
        [
            [float(per_case_mean_cost[case_id].get(algorithm, np.nan)) for case_id in case_ids]
            for algorithm in algorithms
        ],
        dtype=float,
    )
    ensure_dir(output_dir)
    figure_width = max(12, 0.65 * len(case_ids))
    figure, axis = plt.subplots(figsize=(figure_width, 5.6))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(case_ids)))
    axis.set_xticklabels(case_ids, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(algorithms)))
    axis.set_yticklabels(algorithms)
    axis.set_xlabel("Case")
    axis.set_ylabel("Algorithm")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.92)
    colorbar.set_label("Mean system cost")

    best_rows = np.nanargmin(matrix, axis=0)
    for column_index, row_index in enumerate(best_rows):
        axis.add_patch(
            plt.Rectangle(
                (column_index - 0.5, row_index - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="white",
                linewidth=1.4,
            )
        )

    figure.tight_layout()
    figure.savefig(Path(output_dir) / "uav_per_case_cost_heatmap.png", dpi=300)
    plt.close(figure)


def plot_uav_benchmark_overview(analysis_path: str, output_dir: str) -> None:
    analysis = load_json(analysis_path)
    algorithms = _ordered_algorithms(analysis)
    success_rate = analysis.get("success_rate", {})
    average_rank = analysis.get("rank_table", {}).get("average_rank", {})
    collision_rate = analysis.get("collision_rate", {})
    algorithm_summary = analysis.get("algorithm_summary", {})
    per_case_mean_cost = analysis.get("per_case_mean_cost", {})
    case_ids = list(per_case_mean_cost.keys())
    if not algorithms:
        return

    ensure_dir(output_dir)
    figure = plt.figure(figsize=(16, 11))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.25])
    axis_success = figure.add_subplot(grid[0, 0])
    axis_rank = figure.add_subplot(grid[0, 1])
    axis_collision = figure.add_subplot(grid[1, 0])
    axis_heatmap = figure.add_subplot(grid[1, 1])

    axis_success.bar(algorithms, [float(success_rate.get(name, 0.0)) for name in algorithms], color="#4c78a8")
    axis_success.set_ylim(0.0, 1.05)
    axis_success.set_ylabel("Strict success rate")
    axis_success.tick_params(axis="x", rotation=25)
    axis_success.grid(axis="y", alpha=0.2)

    rank_values = [float(average_rank.get(name, np.nan)) for name in algorithms]
    axis_rank.bar(algorithms, rank_values, color="#f58518")
    axis_rank.set_ylabel("Average rank")
    axis_rank.tick_params(axis="x", rotation=25)
    axis_rank.grid(axis="y", alpha=0.2)

    collision_free_values = [float(collision_rate.get(name, {}).get("collision_free_rate", 0.0)) for name in algorithms]
    mean_cost_values = [float(algorithm_summary.get(name, {}).get("mean_system_cost", 0.0)) for name in algorithms]
    axis_collision.bar(np.arange(len(algorithms)) - 0.18, collision_free_values, width=0.36, color="#72b7b2", label="Collision-free rate")
    axis_collision.set_ylim(0.0, 1.05)
    axis_collision.set_ylabel("Rate")
    axis_collision.set_xticks(np.arange(len(algorithms)))
    axis_collision.set_xticklabels(algorithms, rotation=25, ha="right")
    axis_collision.grid(axis="y", alpha=0.2)
    twin_axis = axis_collision.twinx()
    twin_axis.plot(np.arange(len(algorithms)), mean_cost_values, color="#e45756", marker="o", linewidth=2.0, label="Mean system cost")
    twin_axis.set_ylabel("Mean system cost")
    lines_left, labels_left = axis_collision.get_legend_handles_labels()
    lines_right, labels_right = twin_axis.get_legend_handles_labels()
    axis_collision.legend(lines_left + lines_right, labels_left + labels_right, frameon=False, loc="upper center")

    if case_ids:
        matrix = np.asarray(
            [
                [float(per_case_mean_cost[case_id].get(algorithm, np.nan)) for case_id in case_ids]
                for algorithm in algorithms
            ],
            dtype=float,
        )
        heatmap = axis_heatmap.imshow(matrix, aspect="auto", cmap="viridis")
        axis_heatmap.set_xticks(np.arange(len(case_ids)))
        axis_heatmap.set_xticklabels(case_ids, rotation=45, ha="right")
        axis_heatmap.set_yticks(np.arange(len(algorithms)))
        axis_heatmap.set_yticklabels(algorithms)
        axis_heatmap.set_xlabel("Case")
        axis_heatmap.set_ylabel("Algorithm")
        colorbar = figure.colorbar(heatmap, ax=axis_heatmap, shrink=0.92)
        colorbar.set_label("Mean system cost")

    figure.suptitle("UAV benchmark overview", fontsize=16, y=0.98)
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))
    figure.savefig(Path(output_dir) / "uav_benchmark_overview.png", dpi=300)
    plt.close(figure)


def _find_scenario(manifest_path: str, case_id: str) -> Dict[str, Any]:
    manifest = load_json(manifest_path)
    for scenario in manifest.get("scenarios", []):
        if scenario.get("case_id") == case_id:
            return scenario
    raise ValueError(f"Scenario {case_id} not found in manifest {manifest_path}")


def _normalize_path_history(record: Dict[str, Any]) -> tuple[List[List[List[float]]], List[float]]:
    artifacts = record.get("artifacts", {})
    path_history = artifacts.get("best_path_history") or []
    final_path = artifacts.get("path_points") or []
    convergence = [float(value) for value in record.get("convergence_curve", [])]
    if not path_history and final_path:
        path_history = [final_path]
    if not convergence and record.get("best_objective") is not None:
        convergence = [float(record["best_objective"])]
    if not path_history:
        raise ValueError("Record does not contain path history or final path data")
    if len(path_history) < len(convergence):
        path_history.extend([path_history[-1]] * (len(convergence) - len(path_history)))
    elif len(convergence) < len(path_history):
        convergence.extend([convergence[-1]] * (len(path_history) - len(convergence)))
    return path_history, convergence


def _draw_obstacles(ax: Any, scenario: Dict[str, Any]) -> None:
    obstacles = scenario.get("obstacles", [])
    if not obstacles:
        return
    azimuth = np.linspace(0.0, 2.0 * np.pi, 24)
    elevation = np.linspace(0.0, np.pi, 12)
    azimuth_grid, elevation_grid = np.meshgrid(azimuth, elevation)
    for obstacle in obstacles:
        if obstacle.get("type") != "sphere":
            continue
        center = np.asarray(obstacle.get("center", [0.0, 0.0, 0.0]), dtype=float)
        radius = float(obstacle.get("radius", 0.0))
        x_grid = center[0] + radius * np.cos(azimuth_grid) * np.sin(elevation_grid)
        y_grid = center[1] + radius * np.sin(azimuth_grid) * np.sin(elevation_grid)
        z_grid = center[2] + radius * np.cos(elevation_grid)
        ax.plot_wireframe(x_grid, y_grid, z_grid, color="0.6", linewidth=0.4, alpha=0.35)


def plot_uav_dynamic_run(manifest_path: str, record_path: str, output_path: str) -> None:
    record = load_json(record_path)
    scenario = _find_scenario(manifest_path, str(record.get("case_id", "")))
    path_history, convergence = _normalize_path_history(record)
    final_path = np.asarray(record.get("artifacts", {}).get("path_points", path_history[-1]), dtype=float)
    start = np.asarray(scenario["start"], dtype=float)
    goal = np.asarray(scenario["goal"], dtype=float)
    bounds = scenario.get("bounds", {})

    ensure_dir(str(Path(output_path).parent))
    plt.rcParams["animation.embed_limit"] = 100
    figure = plt.figure(figsize=(13, 6))
    path_axis = figure.add_subplot(1, 2, 1, projection="3d")
    convergence_axis = figure.add_subplot(1, 2, 2)

    x_bounds = bounds.get("x", [min(start[0], goal[0]), max(start[0], goal[0])])
    y_bounds = bounds.get("y", [min(start[1], goal[1]), max(start[1], goal[1])])
    z_bounds = bounds.get("z", [min(start[2], goal[2]), max(start[2], goal[2])])
    path_axis.set_xlim(float(x_bounds[0]), float(x_bounds[1]))
    path_axis.set_ylim(float(y_bounds[0]), float(y_bounds[1]))
    path_axis.set_zlim(float(z_bounds[0]), float(z_bounds[1]))
    path_axis.set_xlabel("X")
    path_axis.set_ylabel("Y")
    path_axis.set_zlabel("Z")
    path_axis.view_init(elev=28, azim=-58)
    path_axis.scatter(start[0], start[1], start[2], color="tab:green", s=55)
    path_axis.scatter(goal[0], goal[1], goal[2], color="tab:blue", s=55)
    if len(final_path) >= 2:
        path_axis.plot(final_path[:, 0], final_path[:, 1], final_path[:, 2], linestyle="--", color="0.7", linewidth=1.5)
    _draw_obstacles(path_axis, scenario)
    dynamic_path, = path_axis.plot([], [], [], color="tab:red", linewidth=2.6)

    convergence_axis.set_xlim(1, max(1, len(convergence)))
    convergence_axis.set_ylim(min(convergence) * 0.95, max(convergence) * 1.05)
    convergence_axis.set_xlabel("Generation")
    convergence_axis.set_ylabel("Best cost")
    convergence_axis.grid(alpha=0.25)
    convergence_line, = convergence_axis.plot([], [], color="tab:orange", linewidth=2.2)
    status_text = convergence_axis.text(0.03, 0.97, "", transform=convergence_axis.transAxes, va="top")

    def update(frame_index: int) -> list[Any]:
        current_path = np.asarray(path_history[frame_index], dtype=float)
        dynamic_path.set_data(current_path[:, 0], current_path[:, 1])
        dynamic_path.set_3d_properties(current_path[:, 2])
        x_values = np.arange(1, frame_index + 2)
        y_values = convergence[: frame_index + 1]
        convergence_line.set_data(x_values, y_values)
        status_text.set_text(
            f"case={record.get('case_id')}  gen={frame_index + 1}/{len(path_history)}\n"
            f"best_cost={y_values[-1]:.3f}  api_calls={record.get('llm_stats', {}).get('success_calls', 0)}"
        )
        path_axis.set_title(f"{record.get('algorithm', 'UAV')} best path")
        convergence_axis.set_title("Convergence")
        return [dynamic_path, convergence_line, status_text]

    animation = FuncAnimation(
        figure,
        update,
        frames=len(path_history),
        interval=900,
        blit=False,
        repeat=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=1))
    plt.close(figure)


def build_uav_dynamic_overview_page(manifest_path: str, raw_dir: str, gif_dir: str, output_html: str) -> None:
    manifest = load_json(manifest_path)
    raw_root = Path(raw_dir)
    gif_root = Path(gif_dir)
    output_path = Path(output_html)
    ensure_dir(output_path.parent)

    scenario_order = [scenario["case_id"] for scenario in manifest.get("scenarios", [])]
    records_by_case: Dict[str, Dict[str, Any]] = {}
    for record_path in raw_root.glob("*.json"):
        record = load_json(record_path)
        records_by_case[str(record.get("case_id", record_path.stem))] = record

    cards: List[str] = []
    system_costs: List[float] = []
    runtimes: List[float] = []
    feasible_count = 0
    api_count = 0
    total_generations = 0

    for case_id in scenario_order:
        record = records_by_case.get(case_id)
        if not record:
            continue
        gif_name = f"{case_id}_dynamic.gif"
        gif_path = gif_root / gif_name
        metrics = record.get("metrics", {})
        runtime = record.get("runtime", {})
        llm_stats = record.get("llm_stats", {})
        convergence = record.get("convergence_curve", [])
        start_cost = float(convergence[0]) if convergence else float(record.get("best_objective", 0.0))
        end_cost = float(record.get("best_objective", 0.0))
        system_costs.append(end_cost)
        runtimes.append(float(runtime.get("wall_clock_sec", 0.0)))
        feasible_count += int(bool(record.get("feasible", False)))
        api_count += int(bool(llm_stats.get("api_used", False)))
        total_generations += max(len(convergence), len(record.get("artifacts", {}).get("best_path_history", [])))
        cards.append(
            f"""
            <section class="card">
              <div class="card-header">
                <div>
                  <h2>{html.escape(case_id)}</h2>
                  <p>{html.escape(str(record.get("level", "")).upper())} · {html.escape(str(record.get("algorithm", "")))}</p>
                </div>
                <a class="raw-link" href="raw/{html.escape(raw_root.joinpath(f'uav_{case_id}_L-ADE-Sniper_r00.json').name)}">JSON</a>
              </div>
              <img src="animations/{html.escape(gif_name)}" alt="{html.escape(case_id)} dynamic UAV run">
              <div class="metrics">
                <span>Cost {end_cost:.3f}</span>
                <span>Start {start_cost:.3f}</span>
                <span>Length {float(metrics.get("path_length", 0.0)):.3f}</span>
                <span>Smooth {float(metrics.get("smoothness_cost", 0.0)):.3f}</span>
                <span>Alt Δ {float(metrics.get("altitude_variation", 0.0)):.3f}</span>
                <span>Collision {int(metrics.get("collision_count", 0))}</span>
                <span>Runtime {float(runtime.get("wall_clock_sec", 0.0)):.2f}s</span>
                <span>API {int(llm_stats.get("success_calls", 0))}</span>
              </div>
            </section>
            """
        )

    average_cost = sum(system_costs) / len(system_costs) if system_costs else 0.0
    average_runtime = sum(runtimes) / len(runtimes) if runtimes else 0.0
    average_generations = total_generations / len(cards) if cards else 0.0
    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UAV Dynamic Overview</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #0b1020;
      --panel: #111831;
      --text: #edf2ff;
      --muted: #a9b4d0;
      --accent: #6ea8fe;
      --border: rgba(255,255,255,0.08);
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: linear-gradient(180deg, #0b1020 0%, #111831 100%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero {{
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 24px;
      margin-bottom: 22px;
      box-shadow: 0 18px 60px rgba(0,0,0,0.22);
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 34px;
    }}
    .sub {{
      color: var(--muted);
      margin: 0 0 18px 0;
      line-height: 1.55;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .stat {{
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px 16px;
    }}
    .stat strong {{
      display: block;
      font-size: 22px;
      margin-bottom: 4px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 12px 40px rgba(0,0,0,0.2);
    }}
    .card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 18px 8px 18px;
    }}
    .card-header h2 {{
      margin: 0;
      font-size: 20px;
    }}
    .card-header p {{
      margin: 4px 0 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .raw-link {{
      color: var(--accent);
      text-decoration: none;
      border: 1px solid rgba(110,168,254,0.4);
      padding: 8px 10px;
      border-radius: 10px;
      font-size: 13px;
      white-space: nowrap;
    }}
    .card img {{
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
    }}
    .metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 14px 18px 18px 18px;
    }}
    .metrics span {{
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 13px;
      color: var(--text);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>UAV 动态运行总览</h1>
      <p class="sub">本页汇总 11 个 UAV 场景的逐代动态运行 GIF。每张卡片展示 3D 路径演化与收敛过程，可直接用于筛选论文中的代表性案例，并与原始 JSON 记录一一对应。</p>
      <div class="stats">
        <div class="stat"><strong>{len(cards)}</strong><span>动态案例数</span></div>
        <div class="stat"><strong>{feasible_count}/{len(cards) if cards else 0}</strong><span>严格可行</span></div>
        <div class="stat"><strong>{api_count}/{len(cards) if cards else 0}</strong><span>真实 API 已使用</span></div>
        <div class="stat"><strong>{average_cost:.3f}</strong><span>平均 system cost</span></div>
        <div class="stat"><strong>{average_runtime:.2f}s</strong><span>平均运行时间</span></div>
        <div class="stat"><strong>{average_generations:.1f}</strong><span>平均动态帧数</span></div>
      </div>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def build_uav_dynamic_paper_figure(manifest_path: str, raw_dir: str, output_path: str) -> None:
    manifest = load_json(manifest_path)
    raw_root = Path(raw_dir)
    records_by_case: Dict[str, Dict[str, Any]] = {}
    for record_path in raw_root.glob("*.json"):
        record = load_json(record_path)
        records_by_case[str(record.get("case_id", record_path.stem))] = record

    scenarios = manifest.get("scenarios", [])
    figure, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes_flat = axes.flatten()

    for axis, scenario in zip(axes_flat, scenarios):
        case_id = str(scenario["case_id"])
        record = records_by_case.get(case_id)
        if not record:
            axis.axis("off")
            continue
        path_history, convergence = _normalize_path_history(record)
        first_path = np.asarray(path_history[0], dtype=float)
        middle_path = np.asarray(path_history[len(path_history) // 2], dtype=float)
        final_path = np.asarray(path_history[-1], dtype=float)
        start = np.asarray(scenario["start"], dtype=float)
        goal = np.asarray(scenario["goal"], dtype=float)
        axis.plot(first_path[:, 0], first_path[:, 1], color="#b8c0d9", linewidth=1.2, label="Initial")
        axis.plot(middle_path[:, 0], middle_path[:, 1], color="#5b8ff9", linewidth=1.4, linestyle="--", label="Middle")
        axis.plot(final_path[:, 0], final_path[:, 1], color="#e74c3c", linewidth=1.8, label="Final")
        axis.scatter(start[0], start[1], color="#2ecc71", s=16)
        axis.scatter(goal[0], goal[1], color="#1f77b4", s=16)
        for obstacle in scenario.get("obstacles", []):
            if obstacle.get("type") != "sphere":
                continue
            center = np.asarray(obstacle.get("center", [0.0, 0.0, 0.0]), dtype=float)
            radius = float(obstacle.get("radius", 0.0))
            circle = plt.Circle((center[0], center[1]), radius, color="0.7", fill=False, linewidth=0.8, alpha=0.5)
            axis.add_patch(circle)
        start_cost = float(convergence[0]) if convergence else float(record.get("best_objective", 0.0))
        final_cost = float(record.get("best_objective", 0.0))
        axis.set_title(f"{case_id} ({scenario.get('level', '')})", fontsize=10)
        axis.text(
            0.02,
            0.03,
            f"{start_cost:.0f} → {final_cost:.0f}",
            transform=axis.transAxes,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.75, "edgecolor": "0.85"},
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.tick_params(labelsize=7)

    for axis in axes_flat[len(scenarios):]:
        axis.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], color="#b8c0d9", linewidth=1.8, label="Initial best path"),
        plt.Line2D([0], [0], color="#5b8ff9", linewidth=1.8, linestyle="--", label="Middle generation"),
        plt.Line2D([0], [0], color="#e74c3c", linewidth=2.0, label="Final best path"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=7, label="Start"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=7, label="Goal"),
    ]
    figure.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.02))
    figure.suptitle("UAV dynamic evolution overview across 11 cases", fontsize=16, y=0.98)
    figure.tight_layout(rect=(0.02, 0.06, 0.98, 0.96))
    output_file = Path(output_path)
    ensure_dir(output_file.parent)
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis")
    parser.add_argument("--output_dir")
    parser.add_argument("--manifest")
    parser.add_argument("--record")
    parser.add_argument("--dynamic_output")
    parser.add_argument("--dynamic_raw_dir")
    parser.add_argument("--dynamic_gif_dir")
    parser.add_argument("--overview_html")
    parser.add_argument("--paper_overview_png")
    args = parser.parse_args()
    if args.analysis and args.output_dir:
        plot_uav_success_rates(args.analysis, args.output_dir)
        plot_uav_average_ranks(args.analysis, args.output_dir)
        plot_uav_feasibility_collision(args.analysis, args.output_dir)
        plot_uav_case_cost_heatmap(args.analysis, args.output_dir)
        plot_uav_benchmark_overview(args.analysis, args.output_dir)
    if args.manifest and args.record and args.dynamic_output:
        plot_uav_dynamic_run(args.manifest, args.record, args.dynamic_output)
    if args.manifest and args.dynamic_raw_dir and args.dynamic_gif_dir and args.overview_html:
        build_uav_dynamic_overview_page(args.manifest, args.dynamic_raw_dir, args.dynamic_gif_dir, args.overview_html)
    if args.manifest and args.dynamic_raw_dir and args.paper_overview_png:
        build_uav_dynamic_paper_figure(args.manifest, args.dynamic_raw_dir, args.paper_overview_png)
    if not (
        (args.analysis and args.output_dir)
        or (args.manifest and args.record and args.dynamic_output)
        or (args.manifest and args.dynamic_raw_dir and args.dynamic_gif_dir and args.overview_html)
        or (args.manifest and args.dynamic_raw_dir and args.paper_overview_png)
    ):
        raise SystemExit("Provide valid arguments for static figures, a single dynamic GIF, or the overview page")


if __name__ == "__main__":
    main()
