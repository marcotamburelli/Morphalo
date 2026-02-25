import json
from pathlib import Path


def write_json_sidecar(out_path: Path, payload: dict) -> Path:
    meta_path = out_path.with_suffix('.json')
    write_json(meta_path, payload=payload)

    return meta_path


def write_json(path: Path, payload: dict):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
