from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from safe_llm_saea.json_utils import NumpyEncoder


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def load_json(input_path: str | Path) -> Dict[str, Any]:
    with Path(input_path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_jsonl(records: Iterable[Dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False, cls=NumpyEncoder))
            fp.write("\n")


def load_jsonl(input_path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(input_path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_run_filename(application: str, case_id: str, algorithm: str, run_id: int) -> str:
    safe_algorithm = algorithm.replace("/", "-").replace(" ", "_")
    return f"{application}_{case_id}_{safe_algorithm}_r{run_id:02d}.json"
