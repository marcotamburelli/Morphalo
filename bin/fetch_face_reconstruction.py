"""Download the three external 3DDFA-V3 assets into the Hugging Face cache."""
from argparse import ArgumentParser
from pathlib import Path
import sys


def main() -> None:
    """Resolve CLI options, download required assets, and print their directory."""
    # Support direct execution from a checkout without an installed package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from morphalo.cache.models import resolve_reconstruction_assets
    from morphalo.nodes.preprocess.face_reconstruction import ASSET_REPOSITORY

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '--id', default=ASSET_REPOSITORY,
        help='Compatible asset dataset repository',
    )
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    paths = resolve_reconstruction_assets(repo_id=args.id, revision=args.revision)
    for name in paths:
        print(f'Ready: {name}', flush=True)
    print(f'Cached assets: {next(iter(paths.values())).parent}', flush=True)


if __name__ == '__main__':
    main()
