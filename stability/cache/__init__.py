from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CacheKey:
    # 'sdxl_base', 'controlnet', 'vae', 'depth_model', ...
    kind: str
    ref: str                  # hf id or local path
    device: str               # 'cuda', 'cpu', 'cuda:0', ...
    dtype: str                # 'float16', 'bfloat16', 'float32', ...
    extra: str = ''


class ModelCache:
    _store: Dict[CacheKey, Any] = {}

    @classmethod
    def get(cls, key: CacheKey) -> Optional[Any]:
        return cls._store.get(key)

    @classmethod
    def put(cls, key: CacheKey, value: Any) -> Any:
        cls._store[key] = value
        return value

    @classmethod
    def clear(cls) -> None:
        cls._store.clear()

    @classmethod
    def pop(cls, key: CacheKey) -> None:
        cls._store.pop(key, None)
