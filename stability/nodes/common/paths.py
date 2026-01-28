import time
from pathlib import Path
from typing import Optional


def ensure_out_dir(output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def make_node_output_path(
    *,
    out_dir: Path,
    node_id: str,
    ext: str = 'png',
    seed: Optional[int] = None,
    tag: Optional[str] = None,
) -> Path:
    """
    Generate an output file path for a node execution.

    The path is created under ``out_dir / node_id`` and uses a timestamp-based
    filename. Optional components such as the random seed can be included
    to make the filename more informative while keeping semantics out of
    directory names.

    Parameters
    ----------
    out_dir : Path
        Base output directory of the DAG execution.
    node_id : str
        Identifier of the node producing the output.
    ext : str, optional
        File extension (without leading dot). Default is ``'png'``.
    seed : int, optional
        Optional random seed to include in the filename.
    tag : str, optional
        Optional tag to override the default timestamp-based tag.

    Returns
    -------
    Path
        Full filesystem path where the output file should be written.

    Notes
    -----
    - The directory ``out_dir / node_id`` is created if it does not exist.
    - The default filename format is ``YYYY-MM-DD_HHMMSS[_seedN].<ext>``.
    """
    node_dir = Path(out_dir) / node_id
    node_dir.mkdir(parents=True, exist_ok=True)

    if tag is None:
        tag = time.strftime('%Y-%m-%d_%H%M%S')

    parts = [tag]
    if seed is not None:
        parts.append(f'seed{seed}')

    filename = '_'.join(parts) + f'.{ext.lstrip('.')}'
    return node_dir / filename
