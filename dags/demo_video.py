from pathlib import Path

from stability.dag import DAG
from stability.nodes.file_image import FileImage
from stability.nodes.ltx.img2video import Img2Video
from stability.nodes.ltx.txt2video import Txt2Video
from stability.nodes.ltx.video_image import FileVideo
from stability.nodes.ltx.preprocess import VideoCannyMap, VideoPoseMap

ROOT = Path(__file__).resolve().parents[1]  # dags/ -> root
CONF = ROOT / 'node_conf'
OUT = ROOT / 'outputs' / 'demo_depth_cn'

with DAG('demo_txt2video', out_dir=ROOT / 'outputs' / 'demo_txt2video') as dag:
    # sorgente “foto” da cui estrai la depth map
    out = Txt2Video(id='gen', spec=CONF / 'video' / 'spec_t2v.conf')

    out


with DAG('demo_img2video', out_dir=ROOT / 'outputs' / 'demo_img2video') as dag:
    # sorgente “foto” da cui estrai la depth map
    initial = FileImage(id='guide_img', path='~/images/keyframe.png')
    out = Img2Video(id='gen', spec=CONF / 'video' / 'spec_i2v.conf')

    initial >> out

with DAG('demo_video_proc', out_dir=ROOT / 'outputs' / 'demo_video_proc') as dag:
    initial = FileVideo(id='guide_img', path='~/videos/source_video.mp4')
    pose = VideoPoseMap(id='pose', spec=CONF /
                        'video' / 'spec_video2pose.conf')
    canny = VideoCannyMap(id='canny', spec=CONF /
                        'video' / 'spec_video2canny.conf')

    initial >> pose
    initial >> canny
