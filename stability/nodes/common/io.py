import json
from pathlib import Path


def write_json_sidecar(out_path: Path, payload: dict) -> Path:
    meta_path = out_path.with_suffix('.json')
    meta_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    return meta_path
