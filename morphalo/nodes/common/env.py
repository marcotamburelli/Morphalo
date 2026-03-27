import os

from morphalo.core.config import (HF_HOME, HF_HUB_CACHE,
                                  HF_HUB_DISABLE_TELEMETRY)


def setup_env():
    os.environ['HF_HOME'] = HF_HOME
    os.environ['HF_HUB_CACHE'] = HF_HUB_CACHE
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = HF_HUB_DISABLE_TELEMETRY
