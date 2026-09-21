from __future__ import annotations

from typing import Any

import pytest

from morphalo.dag import DAG
from morphalo.nodes.preprocess.any_crop import AnyCrop
from morphalo.nodes.preprocess.face_crop import FaceCrop
from morphalo.nodes.preprocess.img_aux_map import ImgAuxMap
from morphalo.nodes.preprocess.subject_crop import SubjectCrop
from morphalo.nodes.evaluate.face_scorer import FaceScorer
from morphalo.nodes.evaluate.person_scorer import PersonScorer
from morphalo.nodes.evaluate.prompt_scorer import PromptScorer
from morphalo.nodes.face_id_embed_image import FaceIdEmbedImage
from morphalo.nodes.foundation.flux2_klein import Flux2Klein
from morphalo.nodes.foundation.omnigen import OmniGen
from morphalo.nodes.foundation.qwen_image import QwenImage
from morphalo.nodes.foundation.qwen_image_edit import QwenImageEdit
from morphalo.nodes.foundation.qwen_image_edit_plus import QwenImageEditPlus
from morphalo.nodes.foundation.qwen_image_inpaint import QwenImageInpaint
from morphalo.nodes.img2img import Img2Img
from morphalo.nodes.inpaint import Inpaint
from morphalo.nodes.ltx.img2video import Img2Video
from morphalo.nodes.ltx.preprocess.video_aux import VideoAuxMap
from morphalo.nodes.ltx.preprocess.video_depth_map import VideoDepthMap
from morphalo.nodes.ltx.preprocess.video_pose_map import VideoPoseMap
from morphalo.nodes.ltx.txt2video import Txt2Video
from morphalo.nodes.prompt import Prompt
from morphalo.nodes.txt2img import Txt2Img


MODEL_DEVICE_NODES = [
    Txt2Img,
    Img2Img,
    Inpaint,
    Txt2Video,
    Img2Video,
    OmniGen,
    PromptScorer,
    FaceScorer,
    AnyCrop,
    SubjectCrop,
    VideoDepthMap,
]


@pytest.mark.parametrize('node_cls', MODEL_DEVICE_NODES)
@pytest.mark.parametrize(
    ('device', 'expected'),
    [
        ('cpu', False),
        ('cuda', True),
        ('cuda:0', True),
    ],
)
def test_model_device_nodes_follow_configured_device(
    tmp_path,
    node_cls: type,
    device: str,
    expected: bool,
):
    with DAG(f'{node_cls.__name__.lower()}_{device.replace(":", "_")}', tmp_path):
        node = node_cls(
            name='node',
            spec={'model': {'device': device}},
        )

    assert node.uses_cuda is expected


def test_face_id_embed_defaults_to_cpu(tmp_path):
    with DAG('face_id_devices', tmp_path):
        default_node = FaceIdEmbedImage(name='default')
        cuda_node = FaceIdEmbedImage(
            name='cuda',
            spec={'model': {'device': 'cuda:1'}},
        )

    assert default_node.uses_cuda is False
    assert cuda_node.uses_cuda is True


@pytest.mark.parametrize(
    ('spec', 'expected'),
    [
        ({}, False),
        ({'lang': 'eng_Latn', 'device': 'cuda'}, False),
        ({'lang': 'ita_Latn', 'device': 'cpu'}, False),
        ({'lang': 'ita_Latn', 'device': 'cuda'}, True),
    ],
)
def test_prompt_uses_cuda_only_when_translating_on_cuda(
    tmp_path,
    spec: dict[str, Any],
    expected: bool,
):
    with DAG('prompt_device', tmp_path):
        node = Prompt(name='prompt', spec=spec)

    assert node.uses_cuda is expected


@pytest.mark.parametrize('node_cls', [ImgAuxMap, VideoAuxMap])
@pytest.mark.parametrize(
    ('processor', 'device', 'expected'),
    [
        ('canny', 'cuda', False),
        ('depth_midas', 'cpu', False),
        ('depth_midas', 'cuda', True),
        ('openpose_full', 'cuda:0', True),
    ],
)
def test_aux_nodes_require_checkpoint_and_cuda_device(
    tmp_path,
    node_cls: type,
    processor: str,
    device: str,
    expected: bool,
):
    with DAG(f'{node_cls.__name__.lower()}_{processor}_{device}', tmp_path):
        node = node_cls(
            name='node',
            spec={'processor': processor, 'device': device},
        )

    assert node.uses_cuda is expected


@pytest.mark.parametrize(
    ('target', 'device', 'expected'),
    [
        ('face', 'cuda', True),
        ('face', 'cpu', False),
        ('eyes', 'cuda', False),
        ('left-eyebrow', 'cuda:0', False),
    ],
)
def test_face_crop_uses_cuda_only_for_sam_target(
    tmp_path,
    target: str,
    device: str,
    expected: bool,
):
    with DAG(f'face_crop_{target}_{device.replace(":", "_")}', tmp_path):
        node = FaceCrop(
            name='node',
            spec={
                'model': {'device': device},
                'params': {'target': target},
            },
        )

    assert node.uses_cuda is expected


@pytest.mark.parametrize(
    'node_cls',
    [QwenImage, QwenImageEdit, QwenImageEditPlus, QwenImageInpaint],
)
def test_qwen_nodes_are_cuda_nodes(tmp_path, node_cls: type):
    with DAG(f'{node_cls.__name__.lower()}_device', tmp_path):
        node = node_cls(name='node')

    assert node.uses_cuda is True


@pytest.mark.parametrize(
    ('device', 'expected'),
    [
        ('cpu', False),
        ('cuda', True),
        ('cuda:0', True),
    ],
)
def test_flux2_klein_follows_configured_device(
    tmp_path,
    device: str,
    expected: bool,
):
    with DAG(f'flux2_klein_{device.replace(":", "_")}', tmp_path):
        node = Flux2Klein(
            name='node',
            spec={'model': {'device': device}},
        )

    assert node.uses_cuda is expected


@pytest.mark.parametrize('node_cls', [PersonScorer, VideoPoseMap])
def test_mediapipe_only_nodes_keep_default_cpu_contract(tmp_path, node_cls: type):
    with DAG(f'{node_cls.__name__.lower()}_device', tmp_path):
        node = node_cls(
            name='node',
            spec={'model': {'device': 'cuda'}},
        )

    assert node.uses_cuda is False
