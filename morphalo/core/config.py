from configparser import ConfigParser
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.ini"

config = ConfigParser()
config.read(CONFIG_PATH)


def _optional_project_path(section: str, key: str) -> Optional[str]:
    value = config.get(section, key, fallback=None)
    if value is None or not value.strip():
        return None
    return str(PROJECT_ROOT / value)


def _optional_value(section: str, key: str) -> Optional[str]:
    value = config.get(section, key, fallback=None)
    if value is None or not value.strip():
        return None
    return value


HF_HOME = _optional_project_path("huggingface", "hf_home")
HF_HUB_CACHE = _optional_project_path("huggingface", "hf_hub_cache")
HF_HUB_DISABLE_TELEMETRY = _optional_value(
    "huggingface",
    "disable_telemetry",
)
HF_HUB_OFFLINE = _optional_value("huggingface", "offline")

MORPHALO_CUDA_CHUNK_SIZE = config.getint(
    "morphalo",
    "cuda_chunk_size",
    fallback=8,
)
