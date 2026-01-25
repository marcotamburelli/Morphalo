from pathlib import Path

from stability.dag import DAG
from stability.nodes.file_image import FileImage
from stability.nodes.ltx.file_video import FileVideo
from stability.nodes.ltx.img2video import Img2Video
from stability.nodes.ltx.preprocess import *
from stability.nodes.ltx.txt2video import Txt2Video

ROOT = Path(__file__).resolve().parents[1]  # dags/ -> root
CONF = ROOT / 'node_conf'

with DAG('demo_txt2video', out_dir=ROOT / 'outputs' / 'demo_txt2video') as dag:
    out = Txt2Video(id='gen', spec=CONF / 'video' / 'spec_t2v.conf')

    out


with DAG('demo_img2video', out_dir=ROOT / 'outputs' / 'demo_img2video') as dag:
    initial_video = FileImage(id='guide_img', path='~/images/keyframe.png')
    out = Img2Video(id='gen', spec=CONF / 'video' / 'spec_i2v.conf')

    initial_video >> out

with DAG('demo_video_proc', out_dir=ROOT / 'outputs' / 'demo_video_proc') as dag:
    initial_video = FileVideo(id='guide_img', path='~/videos/source_video.mp4')
    depth = VideoPoseMap(
        id='pose',
        spec=CONF / 'video' / 'spec_video2pose.conf'
    )
    canny = VideoCannyMap(
        id='canny',
        spec=CONF / 'video' / 'spec_video2canny.conf'
    )
    depth = VideoDepthMap(
        id='depth',
        spec=CONF / 'video' / 'spec_video2depth.conf'
    )

    initial_video >> depth
    initial_video >> canny
    initial_video >> depth

with DAG('demo_video_ic_lora_canny', out_dir=ROOT / 'outputs' / 'demo_video_canny') as dag:
    initial_video = FileVideo(
        id='initial_video',
        path='~/videos/source_video.mp4'
    )

    canny = VideoCannyMap(
        id='canny',
        spec=CONF / 'video' / 'spec_video2canny.conf'
    )

    out = Txt2Video(
        id='out',
        spec=CONF / 'video' / 'spec_t2v-ic-lora-canny.conf'
    )

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

    depth = VideoPoseMap(
        id='pose',
        spec=CONF / 'video' / 'spec_video2pose.conf'
    )

    out = Txt2Video(
        id='out',
        spec=CONF / 'video' / 'spec_t2v-ic-lora-pose.conf'
    )

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

    depth = VideoDepthMap(
        id='depth',
        spec=CONF / 'video' / 'spec_video2depth.conf'
    )

    out = Txt2Video(
        id='out',
        spec=CONF / 'video' / 'spec_t2v-ic-lora-depth.conf'
    )

    initial_video >> depth >> out.ic_lora(
        model_id='Lightricks/LTX-Video-ICLoRA-depth-13b-0.9.7',
        weight_name='ltxv-097-ic-lora-depth-control-diffusers.safetensors',
        adapter_name='depth',
        adapter_weight=1.2
    )
