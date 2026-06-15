from __future__ import annotations

import subprocess
import sys
import textwrap


def test_node_imports_do_not_probe_cuda_in_parent_process() -> None:
    script = textwrap.dedent(
        """
        import torch

        calls = []
        for name in ('is_available', 'device_count', 'current_device'):
            original = getattr(torch.cuda, name)

            def wrapper(
                *args,
                __name=name,
                __original=original,
                **kwargs,
            ):
                calls.append(__name)
                return __original(*args, **kwargs)

            setattr(torch.cuda, name, wrapper)

        import morphalo.cache.models
        import morphalo.cache.ltx_models
        import morphalo.nodes
        import morphalo.nodes.foundation
        import morphalo.nodes.ltx

        if calls:
            raise AssertionError(f'CUDA probed during parent imports: {calls}')
        """
    )

    subprocess.run(
        [sys.executable, '-c', script],
        check=True,
        capture_output=True,
        text=True,
    )
