from pathlib import Path
import os
from pyhocon import ConfigFactory


def load_hocon_spec(path: str) -> dict:
    """
    Load a HOCON configuration file and return a plain Python dict
    with all includes and substitutions resolved.
    """
    path = Path(path).expanduser().resolve()

    conf = ConfigFactory.parse_file(path)

    # Convert to plain dict (no ConfigTree)
    spec = conf.as_plain_ordered_dict()

    # Normalize common fields
    _normalize_model(spec)
    _normalize_seed(spec)
    _normalize_prompt(spec)

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

def _normalize_prompt(spec:dict, joiner: str = '\n') -> str:
    prompt = spec.get('prompt')
    if prompt is None:
        return

    if isinstance(prompt, str):
        spec['prompt'] = prompt.strip()

    if isinstance(prompt, list):
        spec['prompt'] = joiner.join(prompt).strip()
