from configparser import ConfigParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.ini"

config = ConfigParser()
config.read(CONFIG_PATH)

HF_HOME = str(
    PROJECT_ROOT / config["huggingface"].get("hf_home", "hf_cache/cache")
)
HF_HUB_CACHE = str(
    PROJECT_ROOT /
    config["huggingface"].get("hf_hub_cache", "hf_cache/hub/cache")
)
HF_HUB_DISABLE_TELEMETRY = config["huggingface"]\
    .get("disable_telemetry", "0")
HF_HUB_OFFLINE = config["huggingface"]\
    .get("offline", "0")

DEVICE = config["torch"].get("device", "cuda")
DTYPE = config["torch"].get("dtype", "float16")
