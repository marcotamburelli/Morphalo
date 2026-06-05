'''
Video demos (WIP / temporary)

This module contains a few minimal LTX-Video DAGs.

Status
------
Video support is still work-in-progress in this repo. Expect breaking changes,
API churn, and incomplete features. These demos are provided mainly to document
the current wiring patterns.

DAGs included
-------------
1) txt2video_min
   Text-to-video from a prompt (no conditioning video).

2) img2video_min
   Image-to-video from a single keyframe image + prompt.

3) txt2video_ic_lora_canny_min
   Text-to-video conditioned by an IC-LoRA using a Canny-map video extracted
   from an input video.

How to run
----------
    ./bin/run_dag.sh demo.14_video_wip --dag txt2video_min
    ./bin/run_dag.sh demo.14_video_wip --dag img2video_min
    ./bin/run_dag.sh demo.14_video_wip --dag txt2video_ic_lora_canny_min
'''

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage, Prompt
from morphalo.nodes.ltx import FileVideo, Img2Video, Txt2Video
from morphalo.nodes.ltx.preprocess import (VideoCannyMap, VideoDepthMap,
                                           VideoPoseMap)

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Local inputs (not committed)
# -----------------------------------------------------------------------------
KEYFRAME_IMG = '~/images/keyframe.png'
SOURCE_VIDEO = '~/videos/source_video.mp4'


# -----------------------------------------------------------------------------
# Shared prompt bundle (keep consistent across demos for comparability)
# -----------------------------------------------------------------------------
PROMPT_SPEC = {
    'lang': 'eng_Latn',
    'prompt': [
        'A young silver-haired mage standing in a garden at dusk.',
        'He slowly raises one hand and begins channeling electrical energy.',
        'Blue-white lightning arcs form between his fingers and swirl around his hand.',
        'Electric sparks flicker and illuminate his face and clothing.',
        'His long silver hair moves slightly in the wind.',
        'The mage prepares to unleash a lightning spell.',
        'Cinematic live-action footage.',
        'Subtle camera movement, slight handheld feel.',
        'Natural lighting with blue highlights from the electricity.',
        'Shallow depth of field, 35mm cinematic look.',
        'Consistent character appearance across frames.',
        'Smooth natural motion.'
    ],
    'negative_prompt': [
        'static frame',
        'frozen pose',
        'jerky motion',
        'temporal flicker',
        'frame inconsistency',
        'low quality',
        'blurry',
        'bad anatomy',
        'distorted face',
        'warped body',
    ],
}

# -----------------------------------------------------------------------------
# 1) Txt2Video (prompt only)
# -----------------------------------------------------------------------------
with DAG('txt2video_min', out_dir=ROOT / 'outputs' / 'txt2video_min') as dag:
    prompt = Prompt(
        name='prompt',
        spec=PROMPT_SPEC,
    )

    out = Txt2Video(
        name='out',
        spec={
            'model': {
                # Defaults exist, but being explicit is clearer in demos.
                # 'id': 'Lightricks/LTX-Video-0.9.7-dev',
                'device': 'cuda',
                'dtype': 'bf16',
                # Optional:
                # 'upscaler': 'Lightricks/ltxv-spatial-upscaler-0.9.7',
            },
            'params': {
                'width': 512,
                'height': 512,
                'fps': 24,
                'num_frames': 96,
                'steps_main': 50,
                'steps_refine': 20,

                # Guidance knobs (see txt2video.py / resolve_ltx_common in ltx_resolve.py)
                # 'guidance_scale': 3.0,
                # 'guidance_rescale': 0.7,
            },
            'seed': 1234,
        },
    )

    prompt >> out.prompt()


# -----------------------------------------------------------------------------
# 2) Img2Video (single keyframe + prompt)
# -----------------------------------------------------------------------------
with DAG('img2video_min', out_dir=ROOT / 'outputs' / 'img2video_min') as dag:
    keyframe = FileImage(
        name='keyframe',
        path=KEYFRAME_IMG,
    )

    prompt = Prompt(
        name='prompt',
        spec=PROMPT_SPEC,
    )

    out = Img2Video(
        name='out',
        spec={
            'model': {
                'device': 'cuda',
                'dtype': 'bf16',
            },
            'params': {
                'width': 512,
                'height': 512,
                'fps': 24,
                'num_frames': 96,
                'steps_main': 50,
                'steps_refine': 20,

                # If your Img2Video implementation supports an anchor frame index,
                # keep it explicit for reproducibility:
                # 'frame_index': 0,
            },
            'seed': 1234,
        },
    )

    keyframe >> out
    prompt >> out.prompt()


# -----------------------------------------------------------------------------
# Local inputs (not committed)
# -----------------------------------------------------------------------------
SOURCE_VIDEO = '~/videos/source_video.mp4'

# -----------------------------------------------------------------------------
# Shared txt2video base spec (inline)
# -----------------------------------------------------------------------------
T2V_BASE_SPEC = {
    'model': {
        'id': 'Lightricks/LTX-Video-0.9.7-dev',
        'upscaler': 'Lightricks/ltxv-spatial-upscaler-0.9.7',
        'device': 'cuda',
        'dtype': 'bf16',
    },
    'params': {
        'width': 512,
        'height': 512,
        'fps': 24,
        'num_frames': 96,
        'steps_main': 50,
        'steps_refine': 20,
    },
    'seed': 1234,
}


# -----------------------------------------------------------------------------
# IC-LoRA preprocess specs (inline)
# -----------------------------------------------------------------------------
VIDEO2CANNY_SPEC = {
    # Preprocessing / rendering
    'preserve_bg': False,         # debug overlay only; keep False for control inputs
    'detect_resolution': 512,
    'image_resolution': 384,
    'max_frames': -1,             # -1 = full video

    # Canny configuration
    'low_threshold': 80,
    'high_threshold': 160,
    'aperture_size': 3,
    'l2': True,                   # enable L2 norm for gradient magnitude
    'invert': False,              # invert final edge map
    # Optional FPS override:
    # 'fps': None,
}

VIDEO2DEPTH_SPEC = {
    'model': {
        'id': 'Intel/dpt-hybrid-midas',
        'device': 'cuda',
        'autocast': True,
    },
    'detect_resolution': 512,
    'image_resolution': 384,
    'normalize': 'running',
    'invert_depth': False,
    'clip_p_low': 10.0,
    'clip_p_high': 70.0,
    'max_frames': -1,
}

VIDEO2POSE_SPEC = {
    # preprocessing
    'preserve_bg': False,
    'max_frames': -1,
    'detect_resolution': 512,
    'image_resolution': 384,

    # extractor params
    'conf_min_pose': 0.3,
    'conf_min_hand': 0.1,
    'conf_min_face': 0.1,
    'max_poses': 4,
    'max_faces': 4,
    'max_hands': 8,

    # local mediapipe tasks (adjust to your installation)
    'models': {
        'pose_task': '~/models/mediapipe/pose_landmarker_heavy.task',
        'hand_task': '~/models/mediapipe/hand_landmarker.task',
        'face_task': '~/models/mediapipe/face_landmarker.task',
    },
}


# -----------------------------------------------------------------------------
# 1) Txt2Video + IC-LoRA (Canny)
# -----------------------------------------------------------------------------
with DAG(
    'demo_video_ic_lora_canny',
    out_dir=ROOT / 'outputs' / 'demo_video_canny',
) as dag:
    initial_video = FileVideo(
        name='initial_video',
        path=SOURCE_VIDEO,
    )

    prompt = Prompt(
        name='prompt',
        spec=PROMPT_SPEC,
    )

    canny = VideoCannyMap(
        name='canny',
        spec=VIDEO2CANNY_SPEC,
    )

    out = Txt2Video(
        name='out',
        spec=T2V_BASE_SPEC,
    )

    prompt >> out.prompt()
    initial_video >> canny >> out.ic_lora(
        model_id='Lightricks/LTX-Video-ICLoRA-canny-13b-0.9.7',
        weight_name='ltxv-097-ic-lora-canny-control-diffusers.safetensors',
        adapter_name='canny',
        adapter_weight=1.0,
    )


# -----------------------------------------------------------------------------
# 2) Txt2Video + IC-LoRA (Pose)
# -----------------------------------------------------------------------------
with DAG(
    'demo_video_ic_lora_pose',
    out_dir=ROOT / 'outputs' / 'demo_video_pose',
) as dag:
    initial_video = FileVideo(
        name='initial_video',
        path=SOURCE_VIDEO,
    )

    prompt = Prompt(
        name='prompt',
        spec=PROMPT_SPEC,
    )

    pose = VideoPoseMap(
        name='pose',
        spec=VIDEO2POSE_SPEC,
    )

    out = Txt2Video(
        name='out',
        spec=T2V_BASE_SPEC,
    )

    prompt >> out.prompt()
    initial_video >> pose >> out.ic_lora(
        model_id='Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7',
        weight_name='ltxv-097-ic-lora-pose-control-diffusers.safetensors',
        adapter_name='pose',
        adapter_weight=1.0,
    )


# -----------------------------------------------------------------------------
# 3) Txt2Video + IC-LoRA (Depth)
# -----------------------------------------------------------------------------
with DAG(
    'demo_video_ic_lora_depth',
    out_dir=ROOT / 'outputs' / 'demo_video_depth',
) as dag:
    initial_video = FileVideo(
        name='initial_video',
        path=SOURCE_VIDEO,
    )

    prompt = Prompt(
        name='prompt',
        spec=PROMPT_SPEC,
    )

    depth = VideoDepthMap(
        name='depth',
        spec=VIDEO2DEPTH_SPEC,
    )

    out = Txt2Video(
        name='out',
        spec={
            **T2V_BASE_SPEC,
            'params': {
                **T2V_BASE_SPEC['params'],
                # Depth often benefits from a bit more refinement, tweak as desired:
                'steps_main': 50,
                'steps_refine': 20,
            },
        },
    )

    prompt >> out.prompt()
    initial_video >> depth >> out.ic_lora(
        model_id='Lightricks/LTX-Video-ICLoRA-depth-13b-0.9.7',
        weight_name='ltxv-097-ic-lora-depth-control-diffusers.safetensors',
        adapter_name='depth',
        adapter_weight=1.0,
    )
