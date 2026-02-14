from pathlib import Path

from stability.dag import DAG
from stability.nodes.file_image import FileImage
from stability.nodes.ltx.file_video import FileVideo
from stability.nodes.ltx.img2video import Img2Video
from stability.nodes.ltx.preprocess import *
from stability.nodes.ltx.txt2video import Txt2Video
from stability.nodes.prompt import Prompt

ROOT = Path(__file__).resolve().parents[1]  # dags/ -> root
CONF = ROOT / 'node_conf'

with DAG('elven_princess', out_dir=ROOT / 'outputs' / 'elven_princess') as dag:
    prompt = Prompt(
        id='elven_princess',
        spec=CONF / 'jobs' / 'elf_princess_garden_i2i.conf'
    )

    out = Txt2Video(
        id='out',
        spec=[
            CONF / 'video' / 'spec_i2v.conf',
            {'params': {
                'width': 512,
                'height': 512,
                'steps_main': 50,
                'steps_refine': 20
            }}
        ]
    )
    prompt >> out.prompt()


with DAG('elven_princess_2', out_dir=ROOT / 'outputs' / 'elven_princess_2') as dag:
    initial_video = FileImage(id='guide_img', path='~/images/keyframe.png')
    prompt = Prompt(
        id='elven_princess',
        spec=CONF / 'jobs' / 'elf_princess_garden_i2i.conf'
    )

    out = Img2Video(
        id='out',
        spec=[
            CONF / 'video' / 'spec_i2v.conf',
            {'params': {
                'width': 512,
                'height': 512,
                'steps_main': 50,
                'steps_refine': 20
            }}
        ]
    )

    initial_video >> out
    prompt >> out.prompt()

with DAG('demo_video_ic_lora_canny', out_dir=ROOT / 'outputs' / 'demo_video_canny') as dag:
    initial_video = FileVideo(
        id='initial_video',
        path='~/videos/source_video.mp4'
    )
    prompt = Prompt(
        id='elven_princess',
        spec=CONF / 'jobs' / 'elf_princess_garden_i2i.conf'
    )

    canny = VideoCannyMap(
        id='canny',
        spec=CONF / 'video' / 'spec_video2canny.conf'
    )

    out = Txt2Video(
        id='out',
        spec=CONF / 'video' / 'spec_t2v-ic-lora-canny.conf'
    )

    prompt >> out.prompt()
    initial_video >> canny >> out.ic_lora(
        model_id='Lightricks/LTX-Video-ICLoRA-canny-13b-0.9.7',
        weight_name='ltxv-097-ic-lora-canny-control-diffusers.safetensors',
        adapter_name='canny',
        adapter_weight=1.0
    )

with DAG('demo_video_ic_lora_pose', out_dir=ROOT / 'outputs' / 'demo_video_pose') as dag:
    initial_video = FileVideo(
        id='initial_video',
        path='~/videos/source_video.mp4'
    )
    prompt = Prompt(
        id='elven_princess',
        spec=CONF / 'jobs' / 'elf_princess_garden_i2i.conf'
    )

    depth = VideoPoseMap(
        id='pose',
        spec=CONF / 'video' / 'spec_video2pose.conf'
    )

    out = Txt2Video(
        id='out',
        spec=CONF / 'video' / 'spec_t2v-ic-lora-pose.conf'
    )

    prompt >> out.prompt()
    initial_video >> depth >> out.ic_lora(
        model_id='Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7',
        weight_name='ltxv-097-ic-lora-pose-control-diffusers.safetensors',
        adapter_name='pose',
        adapter_weight=1.0
    )

with DAG('demo_video_ic_lora_depth', out_dir=ROOT / 'outputs' / 'demo_video_depth') as dag:
    initial_video = FileVideo(
        id='initial_video',
        path='~/videos/source_video.mp4'
    )
    prompt = Prompt(
        id='elven_princess',
        spec=CONF / 'jobs' / 'elf_princess_garden_i2i.conf'
    )

    depth = VideoDepthMap(
        id='depth',
        spec=CONF / 'video' / 'spec_video2depth.conf'
    )

    out = Txt2Video(
        id='out',
        spec=[
            CONF / 'video' / 'spec_t2v-ic-lora-depth.conf',
            {'params': {
                'steps_main': 50,
                'steps_refine': 20
            }}
        ]
    )

    prompt >> out.prompt()
    initial_video >> depth >> out.ic_lora(
        model_id='Lightricks/LTX-Video-ICLoRA-depth-13b-0.9.7',
        weight_name='ltxv-097-ic-lora-depth-control-diffusers.safetensors',
        adapter_name='depth',
        adapter_weight=1
    )
