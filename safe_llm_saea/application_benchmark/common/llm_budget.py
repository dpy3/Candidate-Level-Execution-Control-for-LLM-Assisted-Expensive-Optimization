from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def init_llm_stats(llm_enabled: bool) -> Dict[str, Any]:
    return {
        "llm_enabled": bool(llm_enabled),
        "total_calls": 0,
        "success_calls": 0,
        "failed_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "token_cost_usd": 0.0,
        "latency_sec_total": 0.0,
        "calls": [],
    }


def record_llm_call(
    llm_stats: Dict[str, Any],
    stage: str,
    success: bool,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    latency_sec: float,
    metadata: Dict[str, Any],
) -> None:
    llm_stats["total_calls"] += 1
    if success:
        llm_stats["success_calls"] += 1
    else:
        llm_stats["failed_calls"] += 1
    llm_stats["prompt_tokens"] += int(prompt_tokens)
    llm_stats["completion_tokens"] += int(completion_tokens)
    llm_stats["token_cost_usd"] += float(cost_usd)
    llm_stats["latency_sec_total"] += float(latency_sec)
    llm_stats["calls"].append(
        {
            "stage": stage,
            "success": bool(success),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "cost_usd": float(cost_usd),
            "latency_sec": float(latency_sec),
            "metadata": deepcopy(metadata),
        }
    )


def finalize_llm_stats(llm_stats: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(llm_stats)
    total_calls = max(1, int(result["total_calls"]))
    result["avg_cost_per_call_usd"] = float(result["token_cost_usd"]) / total_calls
    result["avg_latency_per_call_sec"] = float(result["latency_sec_total"]) / total_calls
    return result
