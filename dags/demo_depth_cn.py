from pathlib import Path

from stability.dag import DAG
from stability.nodes.file_image import FileImage
from stability.nodes.img2img import Img2Img
from stability.nodes.img_helper import DepthMap
from stability.nodes.txt2img import Txt2Img

ROOT = Path(__file__).resolve().parents[1]  # dags/ -> root
CONF = ROOT / 'node_conf'
OUT = ROOT / 'outputs' / 'demo_depth_cn'

with DAG('demo', out_dir=ROOT / 'outputs' / 'demo') as dag:
    # sorgente “foto” da cui estrai la depth map
    guide = FileImage(id='guide_img', path='~/images/elven_princess.png')

    # preprocess: foto -> depth map
    depth = DepthMap(id='depth', device='cuda', size=1024)
    guide >> depth

    # 1) genera un’immagine dal prompt
    gen = Txt2Img(id='gen', spec=CONF / 'jobs' / 'model_pool_t2i.conf')

    # 2) img2img a partire da gen, condizionata dalla depth map
    out = Img2Img(id='out', spec=CONF / 'jobs' /
                  'elf_princess_garden_i2i.conf')

    gen >> out  # init image per img2img
    depth >> out.controlnet.add(
        'diffusers/controlnet-depth-sdxl-1.0',
        conditioning_scale=0.5,
        key='depth1',
    )
