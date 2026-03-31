import gc

import torch


def cleanup_torch_cuda() -> None:
    """
    Perform best-effort Python and CUDA memory cleanup.

    Notes
    -----
    This only releases CUDA memory that is no longer strongly referenced.
    ``torch.cuda.empty_cache()`` releases unoccupied cached memory held by
    the allocator, but it does not free memory still owned by live tensors
    or modules.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
