import gc

import torch


def synchronize_torch_cuda() -> None:
    """Wait for pending CUDA work without initializing CUDA."""
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


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

    if torch.cuda.is_initialized():
        synchronize_torch_cuda()
        torch.cuda.empty_cache()


class CudaPostRunMixin:
    """Synchronize CUDA work after nodes that may execute on CUDA."""

    def post_run(self) -> None:
        if self.uses_cuda:
            synchronize_torch_cuda()

        super().post_run()
