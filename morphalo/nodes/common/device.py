# This module intentionally starts with a single function. Device interpretation
# is a runtime concern shared by otherwise unrelated nodes, while config_resolve
# is responsible only for loading and combining specifications. Keeping these
# concerns separate avoids turning the configuration module into a collection of
# unrelated utilities and leaves a clear home for future helpers such as
# device_type() or normalize_device().

from typing import Any


def is_cuda_device(device: Any) -> bool:
    """Return whether a runtime device string selects CUDA."""
    value = str(device).strip().lower()
    return value == 'cuda' or value.startswith('cuda:')

