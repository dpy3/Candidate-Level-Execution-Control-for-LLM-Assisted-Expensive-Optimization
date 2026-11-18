from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from safe_llm_saea.csa_protocol import candidate_id


EPS = 1e-12
MIN_STEP_SCALE = 0.005
MAX_STEP_SCALE = 0.25


@dataclass(frozen=True)
class DirectionalInstruction:
    direction: tuple[float, ...]
    step_scale: float
    intent: str


def normalize_point(
    point: Sequence[float], bounds: Sequence[Sequence[float]]
) -> np.ndarray:
    value = np.asarray(point, dtype=float)
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    span = np.maximum(EPS, np.asarray([item[1] - item[0] for item in bounds], dtype=float))
    return np.clip(2.0 * (value - lower) / span - 1.0, -1.0, 1.0)


def summarize_de_state(
    population: Sequence[Sequence[float]],
    fitness: Sequence[float],
    bounds: Sequence[Sequence[float]],
    progress: float,
) -> dict[str, Any]:
    x = np.asarray(population, dtype=float)
    y = np.asarray(fitness, dtype=float)
    order = np.argsort(y)
    best = x[order[0]]
    normalized = np.asarray([normalize_point(row, bounds) for row in x])
    best_normalized = normalize_point(best, bounds)
    top = normalized[order[: min(3, len(order))]]
    centroid = np.mean(normalized, axis=0)
    spread = np.std(normalized, axis=0)
    return {
        "backbone": "DE",
        "dimension": int(x.shape[1]),
        "progress": float(progress),
        "incumbent_normalized": best_normalized.tolist(),
        "top_offsets_from_incumbent": (top - best_normalized).tolist(),
        "centroid_offset_from_incumbent": (centroid - best_normalized).tolist(),
        "coordinate_spread": spread.tolist(),
        "diversity_rms": float(np.sqrt(np.mean(spread ** 2))),
    }


def summarize_pso_state(
    positions: Sequence[Sequence[float]],
    values: Sequence[float],
    velocities: Sequence[Sequence[float]],
    bounds: Sequence[Sequence[float]],
    progress: float,
) -> dict[str, Any]:
    x = np.asarray(positions, dtype=float)
    y = np.asarray(values, dtype=float)
    v = np.asarray(velocities, dtype=float)
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    span = np.maximum(EPS, np.asarray([item[1] - item[0] for item in bounds], dtype=float))
    normalized = np.asarray([normalize_point(row, bounds) for row in x])
    best_normalized = normalized[int(np.argmin(y))]
    centroid = np.mean(normalized, axis=0)
    velocity_scaled = v / span
    spread = np.std(normalized, axis=0)
    return {
        "backbone": "PSO",
        "dimension": int(x.shape[1]),
        "progress": float(progress),
        "incumbent_normalized": best_normalized.tolist(),
        "swarm_centroid_offset": (centroid - best_normalized).tolist(),
        "mean_velocity_direction": np.mean(velocity_scaled, axis=0).tolist(),
        "velocity_rms": float(np.sqrt(np.mean(velocity_scaled ** 2))),
        "coordinate_spread": spread.tolist(),
        "diversity_rms": float(np.sqrt(np.mean(spread ** 2))),
    }


def build_directional_prompt(
    state: Mapping[str, Any], proposal_count: int = 3
) -> tuple[str, str, str]:
    backbone = str(state["backbone"])
    dimension = int(state["dimension"])
    if backbone == "DE":
        role = (
            "Use the incumbent-relative offsets, centroid offset, and coordinate "
            "spread to propose locally useful mutation directions for differential evolution."
        )
    elif backbone == "PSO":
        role = (
            "Use the swarm centroid, mean velocity, and coordinate spread to propose "
            "locally useful directions around the current global best for particle swarm optimization."
        )
    else:
        raise ValueError("unsupported directional backbone")
    system = f"""You design bounded search directions for {backbone}, not absolute solutions.
{role}
Return strict JSON only with this schema:
{{"proposals":[{{"direction":[d1,...,d{dimension}],"step_scale":0.05,"intent":"short label"}}]}}
Rules:
- Return exactly {int(proposal_count)} proposals.
- Each direction has exactly {dimension} finite numbers in [-1,1] and is not all zero.
- step_scale is in [{MIN_STEP_SCALE},{MAX_STEP_SCALE}] and represents a fraction of the box width.
- Mix one conservative direction, one population-informed direction, and one diversity-restoring direction.
- Do not output absolute coordinates, objective values, optimizer parameters, explanations, or markdown.
"""
    user = "Normalized host state:\n" + json.dumps(dict(state), separators=(",", ":"))
    prompt_sha256 = hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest()
    return system, user, prompt_sha256


def _json_content(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return json.loads(text.strip())


def parse_directional_response(
    content: str, dimension: int, proposal_count: int = 3
) -> list[DirectionalInstruction]:
    payload = _json_content(content)
    proposals = payload.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != int(proposal_count):
        raise ValueError("directional response has the wrong proposal count")
    parsed = []
    for item in proposals:
        if not isinstance(item, Mapping):
            raise ValueError("directional proposal must be an object")
        direction = np.asarray(item.get("direction"), dtype=float)
        if direction.shape != (int(dimension),) or not np.all(np.isfinite(direction)):
            raise ValueError("directional proposal has invalid dimension or values")
        if np.max(np.abs(direction)) > 1.0 + EPS or np.max(np.abs(direction)) < EPS:
            raise ValueError("direction must be nonzero and bounded by one")
        step = float(item.get("step_scale"))
        if not MIN_STEP_SCALE <= step <= MAX_STEP_SCALE:
            raise ValueError("directional step scale is outside the host contract")
        normalized = direction / max(EPS, float(np.max(np.abs(direction))))
        parsed.append(DirectionalInstruction(
            tuple(float(value) for value in normalized),
            step,
            str(item.get("intent", "unspecified"))[:80],
        ))
    return parsed


def construct_directional_candidates(
    incumbent: Sequence[float],
    instructions: Sequence[DirectionalInstruction],
    bounds: Sequence[Sequence[float]],
) -> tuple[list[np.ndarray], dict[str, list[str]], list[dict[str, Any]]]:
    x = np.asarray(incumbent, dtype=float)
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    upper = np.asarray([item[1] for item in bounds], dtype=float)
    span = upper - lower
    source_id = candidate_id(x)
    candidates = []
    lineage = {}
    records = []
    for index, instruction in enumerate(instructions):
        direction = np.asarray(instruction.direction, dtype=float)
        candidate = np.clip(x + instruction.step_scale * span * direction, lower, upper)
        cid = candidate_id(candidate)
        candidates.append(candidate)
        lineage[cid] = [source_id]
        records.append({
            "candidate_id": cid,
            "source_incumbent_id": source_id,
            "instruction_index": index,
            "direction": direction.tolist(),
            "step_scale": float(instruction.step_scale),
            "intent": instruction.intent,
        })
    return candidates, lineage, records
