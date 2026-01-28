import torch


def cuda_prerun(device: str) -> None:
    """
    Prepare CUDA state for timing/memory measurements.

    Resets peak memory stats and synchronizes the device so subsequent timing
    reflects the work of the current run only.
    """
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def cuda_sync(device: str) -> None:
    """Synchronize CUDA device if running on CUDA."""
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()


def cuda_mem_stats(device: str) -> dict:
    """
    Return CUDA memory statistics in GB, or an empty dict if not on CUDA.
    """
    if torch.device(device).type != 'cuda':
        return {}

    return {
        'allocated_gb': round(torch.cuda.memory_allocated() / 1024**3, 3),
        'reserved_gb': round(torch.cuda.memory_reserved() / 1024**3, 3),
        'peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        'peak_reserved_gb': round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }
