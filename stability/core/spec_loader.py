import os
from pathlib import Path
from typing import Union

from pyhocon import ConfigFactory


def load_hocon_spec(path: Union[str, Path]) -> dict:
    """
    Load a HOCON configuration file and return a plain Python dict
    with all includes and substitutions resolved.

    Parameters
    ----------
    path : str or Path
        Path to the HOCON configuration file.

    Returns
    -------
    dict
        Parsed and normalized configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    """
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f'HOCON spec file not found: {path}')

    if not path.is_file():
        raise ValueError(f'HOCON spec path is not a file: {path}')

    conf = ConfigFactory.parse_file(str(path))

    # Convert to plain dict (no ConfigTree)
    spec = conf.as_plain_ordered_dict()

    # Normalize common fields
    _normalize_model(spec)
    _normalize_seed(spec)

    return spec


def _normalize_model(spec: dict) -> None:
    model = spec.get('model')
    if not model:
        return

    if 'path' in model:
        model['path'] = os.path.expanduser(model['path'])

    if 'dtype' in model:
        model['dtype'] = model['dtype'].lower()


def _normalize_seed(spec: dict) -> None:
    seed = spec.get('seed')
    if seed is None:
        spec['seed'] = 'random'
