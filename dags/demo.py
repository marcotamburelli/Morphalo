from pathlib import Path

from stability.dag import DAG
from stability.nodes.file_image import FileImage
from stability.nodes.img2img import Img2Img
from stability.nodes.img_helper import CannyEdge, DepthMap
from stability.nodes.txt2img import Txt2Img

ROOT = Path(__file__).resolve().parents[1]  # dags/ -> root
CONF = ROOT / 'node_conf'
OUT = ROOT / 'outputs' / 'demo_depth_cn'

with DAG('demo_depth_1', out_dir=ROOT / 'outputs' / 'demo_depth_1') as dag:
    # sorgente “foto” da cui estrai la depth map
    initial = FileImage(id='guide_img', path='~/images/elven_princess.png')

    # preprocess: foto -> depth map
    canny = DepthMap(id='depth', device='cuda', size=1024)
    initial >> canny

    # 1) genera un’immagine dal prompt con la mappa di profondita
    out = Txt2Img(id='gen', spec=CONF / 'jobs' /
                  'elf_princess_garden_i2i.conf')

    canny >> out.controlnet.add(
        'diffusers/controlnet-depth-sdxl-1.0',
        conditioning_scale=0.5,
        key='depth1',
    )

with DAG('demo_depth_2', out_dir=ROOT / 'outputs' / 'demo_depth_2') as dag:
    # sorgente “foto” da cui estrai la depth map
    initial = FileImage(id='guide_img', path='~/images/elven_princess.png')

    # preprocess: foto -> depth map
    canny = DepthMap(id='depth', device='cuda', size=1024)
    initial >> canny

    # 1) genera un’immagine dal prompt
    gen = Txt2Img(id='gen', spec=CONF / 'jobs' / 'model_pool_t2i.conf')

    # 2) img2img a partire da gen, condizionata dalla depth map
    out = Img2Img(id='out', spec=CONF / 'jobs' /
                  'elf_princess_garden_i2i.conf')

    gen >> out  # init image per img2img
    canny >> out.controlnet.add(
        'diffusers/controlnet-depth-sdxl-1.0',
        conditioning_scale=0.5,
        key='depth1',
    )

with DAG('demo_ip_adapter_1', out_dir=ROOT / 'outputs' / 'demo_ip_adapter_1') as dag:
    # sorgente “foto” da cui estrai la depth map
    initial = FileImage(id='guide_img', path='~/images/elven_princess.png')

    # 1) genera un’immagine dal prompt con la mappa di profondita
    out = Txt2Img(id='gen', spec=CONF / 'jobs' /
                  'elf_princess_garden_i2i.conf')

    initial >> out.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter_sdxl.bin',
        scale=0.5,
        key='IPA1',
    )

with DAG('demo_ip_adapter_2', out_dir=ROOT / 'outputs' / 'demo_ip_adapter_2') as dag:
    # sorgente “foto” da cui estrai la depth map
    initial = FileImage(id='guide_img', path='~/images/elven_princess.png')

    # 1) genera un’immagine dal prompt
    gen = Txt2Img(id='gen', spec=CONF / 'jobs' / 'model_pool_t2i.conf')

    # 2) img2img a partire da gen, condizionata dalla immagine iniziale
    out = Img2Img(id='out', spec=CONF / 'jobs' /
                  'elf_princess_garden_i2i.conf')

    gen >> out  # init image per img2img
    initial >> out.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter_sdxl.bin',
        scale=0.5,
        key='IPA1',
    )


# with DAG('demo_ip_adapter_2', out_dir=ROOT / 'outputs' / 'demo_ip_adapter_2') as dag:
#     # sorgente “foto” da cui estrai la depth map
#     initial = FileImage(id='guide_img', path='~/images/elven_princess.png')

#     # 1) genera un’immagine dal prompt
#     gen = Txt2Img(id='gen', spec=CONF / 'jobs' / 'model_pool_t2i.conf')

#     # 2) img2img a partire da gen, condizionata dalla immagine iniziale
#     out = Img2Img(id='out', spec=CONF / 'jobs' /
#                   'elf_princess_garden_i2i.conf')

#     gen >> out  # init image per img2img
#     initial >> out.ip_adapter.add(
#         'h94/IP-Adapter',
#         subfolder='sdxl_models',
#         weight_name='ip-adapter_sdxl.bin',
#         scale=0.5,
#         key='IPA1',
#     )

with DAG('demo_mix', out_dir=ROOT / 'outputs' / 'demo_mix') as dag:
    # sorgente “foto” da cui estrai la depth map
    initial = FileImage(id='guide_img', path='~/images/elven_princess.png')

    # 1) genera un’immagine dal prompt
    gen = Txt2Img(id='gen', spec=CONF / 'jobs' / 'model_pool_t2i.conf')
    # preprocess: foto -> canny edges
    canny = CannyEdge(id='canny')

    gen >> canny
    # 2) img2img a partire da gen, condizionata dalla immagine iniziale
    out = Img2Img(id='out', spec=CONF / 'jobs' /
                  'elf_princess_garden_i2i.conf')

    gen >> out  # init image per img2img
    initial >> out.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter_sdxl.bin',
        scale=0.5,
        key='IPA1',
    )
    canny >> out.controlnet.add(
        'diffusers/controlnet-canny-sdxl-1.0',
        conditioning_scale=0.5,
        key='canny1',
    )
