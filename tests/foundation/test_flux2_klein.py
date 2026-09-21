from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from morphalo.dag import DAG
from morphalo.nodes.foundation import flux2_klein
from morphalo.nodes.foundation.flux2_klein import Flux2Klein


@dataclass
class _FakeResult:
    images: list[Image.Image]


class _FakePipe:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResult(images=[Image.new('RGB', (8, 8), 'white')])


def test_flux2_klein_txt2img_does_not_pass_image(monkeypatch, tmp_path) -> None:
    fake_pipe = _FakePipe()

    monkeypatch.setattr(
        flux2_klein,
        'get_flux2_klein_pipe',
        lambda **kwargs: fake_pipe,
    )

    with DAG('flux2_klein_txt2img', out_dir=tmp_path):
        node = Flux2Klein(
            name='out',
            spec={
                'model': {'device': 'cpu'},
                'prompt': 'A clean product photo of a ceramic cup.',
                'params': {'steps': 7, 'cfg': 3.5},
                'seed': 123,
            },
        )

    out = node.run(output_dir=tmp_path, input={})

    assert out['ok'] is True
    assert fake_pipe.calls
    assert 'image' not in fake_pipe.calls[0]
    assert fake_pipe.calls[0]['prompt'] == 'A clean product photo of a ceramic cup.'
    assert fake_pipe.calls[0]['num_inference_steps'] == 7
    assert fake_pipe.calls[0]['guidance_scale'] == 3.5


def test_flux2_klein_cfg_takes_precedence_over_guidance_scale(monkeypatch, tmp_path) -> None:
    fake_pipe = _FakePipe()

    monkeypatch.setattr(
        flux2_klein,
        'get_flux2_klein_pipe',
        lambda **kwargs: fake_pipe,
    )

    with DAG('flux2_klein_cfg_alias', out_dir=tmp_path):
        node = Flux2Klein(
            name='out',
            spec={
                'model': {'device': 'cpu'},
                'prompt': 'A clean product photo.',
                'params': {'cfg': 2.0, 'guidance_scale': 7.0},
                'seed': 123,
            },
        )

    out = node.run(output_dir=tmp_path, input={})

    assert out['ok'] is True
    assert fake_pipe.calls[0]['guidance_scale'] == 2.0
    assert out['params']['guidance_scale'] == 2.0


def test_flux2_klein_long_side_requires_input_image(tmp_path) -> None:
    with DAG('flux2_klein_long_side_no_image', out_dir=tmp_path):
        node = Flux2Klein(
            name='out',
            spec={
                'model': {'device': 'cpu'},
                'prompt': 'A clean product photo.',
                'params': {'long_side': 64},
                'seed': 123,
            },
        )

    try:
        node.run(output_dir=tmp_path, input={})
    except ValueError as exc:
        assert "params 'long_side' requires at least one input image" in str(exc)
    else:
        raise AssertionError('Expected Flux2Klein to reject long_side without images.')


def test_flux2_klein_long_side_uses_first_image(monkeypatch, tmp_path) -> None:
    fake_pipe = _FakePipe()

    monkeypatch.setattr(
        flux2_klein,
        'get_flux2_klein_pipe',
        lambda **kwargs: fake_pipe,
    )

    with DAG('flux2_klein_long_side_image', out_dir=tmp_path):
        node = Flux2Klein(
            name='out',
            spec={
                'model': {'device': 'cpu'},
                'prompt': 'Use the reference composition.',
                'params': {'long_side': 64, 'width': 512, 'height': 512},
                'seed': 123,
            },
        )

    image = Image.new('RGB', (20, 10), 'white')
    bundle = type(
        'Bundle',
        (),
        {
            'images': [image],
            'metadata': [],
        },
    )()
    node.build_image_sequence_bundle = lambda input: bundle

    out = node.run(output_dir=tmp_path, input={})

    assert out['ok'] is True
    assert fake_pipe.calls[0]['image'] is image
    assert fake_pipe.calls[0]['width'] == 64
    assert fake_pipe.calls[0]['height'] == 32
    assert out['params']['long_side'] == 64
