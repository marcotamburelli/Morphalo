import os

from morphalo.core.config import (HF_HOME, HF_HUB_CACHE,
                                  HF_HUB_DISABLE_TELEMETRY, HF_HUB_OFFLINE)


def setup_env():
    values = {
        'HF_HOME': HF_HOME,
        'HF_HUB_CACHE': HF_HUB_CACHE,
        'HF_HUB_DISABLE_TELEMETRY': HF_HUB_DISABLE_TELEMETRY,
        'HF_HUB_OFFLINE': HF_HUB_OFFLINE,
    }
    for key, value in values.items():
        if value is not None:
            os.environ[key] = value
