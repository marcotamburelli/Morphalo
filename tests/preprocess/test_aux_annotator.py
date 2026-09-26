import pytest

from morphalo.nodes.preprocess.utils import aux_annotator


class _StatelessAnnotator:
    pass


def test_build_aux_annotator_constructs_stateless_processor(monkeypatch):
    monkeypatch.setattr(
        aux_annotator,
        'MODELS',
        {'canny': {'class': _StatelessAnnotator, 'checkpoint': False}},
    )

    annotator = aux_annotator.build_aux_annotator('canny', device='cpu')

    assert isinstance(annotator, _StatelessAnnotator)


def test_build_aux_annotator_uses_cache_for_checkpoint(monkeypatch):
    expected = object()
    calls = []
    monkeypatch.setattr(
        aux_annotator,
        'MODELS',
        {'depth_midas': {'class': _StatelessAnnotator, 'checkpoint': True}},
    )

    def _get_annotator(**kwargs):
        calls.append(kwargs)

        return expected

    monkeypatch.setattr(
        aux_annotator,
        'get_controlnet_aux_annotator',
        _get_annotator,
    )

    result = aux_annotator.build_aux_annotator(
        'depth_midas',
        device='cuda',
    )

    assert result is expected
    assert calls == [{
        'processor': 'depth_midas',
        'cls': _StatelessAnnotator,
        'device': 'cuda',
    }]


def test_build_aux_annotator_rejects_unknown_processor(monkeypatch):
    monkeypatch.setattr(aux_annotator, 'MODELS', {})

    with pytest.raises(ValueError, match='Unknown processor=missing'):
        aux_annotator.build_aux_annotator('missing', device='cpu')


def test_build_aux_annotator_rejects_bundled_dwpose(monkeypatch):
    monkeypatch.setattr(
        aux_annotator,
        'MODELS',
        {'dwpose': {'class': _StatelessAnnotator, 'checkpoint': True}},
    )

    with pytest.raises(ValueError, match='dwpose is not available'):
        aux_annotator.build_aux_annotator('dwpose', device='cpu')
