import pytest

from morphalo.dag import DAG
from morphalo.nodes.common import cuda_mem
from morphalo.nodes.foundation import omnigen
from morphalo.nodes.foundation import qwen_image


class _PostRunRecorder:
    def __init__(self, *, uses_cuda: bool, events: list[str]) -> None:
        self._uses_cuda = uses_cuda
        self.events = events

    @property
    def uses_cuda(self) -> bool:
        return self._uses_cuda

    def post_run(self) -> None:
        self.events.append('parent')


class _CudaPostRunNode(cuda_mem.CudaPostRunMixin, _PostRunRecorder):
    pass


def test_synchronize_does_not_initialize_cuda(monkeypatch):
    events = []
    monkeypatch.setattr(
        cuda_mem.torch.cuda,
        'is_initialized',
        lambda: False,
    )
    monkeypatch.setattr(
        cuda_mem.torch.cuda,
        'synchronize',
        lambda: events.append('synchronize'),
    )

    cuda_mem.synchronize_torch_cuda()

    assert events == []


@pytest.mark.parametrize(
    ('uses_cuda', 'expected'),
    [
        (False, ['parent']),
        (True, ['synchronize', 'parent']),
    ],
)
def test_cuda_post_run_mixin_synchronizes_only_cuda_nodes(
    monkeypatch,
    uses_cuda,
    expected,
):
    events = []
    monkeypatch.setattr(
        cuda_mem,
        'synchronize_torch_cuda',
        lambda: events.append('synchronize'),
    )

    node = _CudaPostRunNode(uses_cuda=uses_cuda, events=events)
    node.post_run()

    assert events == expected


def test_foundation_cleanup_runs_when_eviction_fails(monkeypatch, tmp_path):
    events = []

    monkeypatch.setattr(
        qwen_image,
        'synchronize_torch_cuda',
        lambda: events.append('synchronize'),
    )
    monkeypatch.setattr(
        qwen_image,
        'cleanup_torch_cuda',
        lambda: events.append('cleanup'),
    )

    def fail_eviction(**kwargs):
        events.append('evict')
        raise RuntimeError('planned eviction failure')

    monkeypatch.setattr(qwen_image, 'evict_qwen_image', fail_eviction)

    with DAG('foundation_post_run', out_dir=tmp_path):
        node = qwen_image.QwenImage(
            name='qwen',
            evict_after_run=True,
        )

    with pytest.raises(RuntimeError, match='planned eviction failure'):
        node.post_run()

    assert events == ['synchronize', 'evict', 'cleanup']


def test_cpu_foundation_node_does_not_synchronize(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        omnigen,
        'synchronize_torch_cuda',
        lambda: events.append('synchronize'),
    )

    with DAG('cpu_foundation_post_run', out_dir=tmp_path):
        node = omnigen.OmniGen(
            name='omnigen',
            spec={'model': {'device': 'cpu'}},
        )

    node.post_run()

    assert events == []
