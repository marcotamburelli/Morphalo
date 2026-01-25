from pathlib import Path

from stability.dag import DAG
from stability.nodes.face_id_embed_image import FaceIdEmbedImage
from stability.nodes.file_image import FileImage
from stability.nodes.img2img import Img2Img
from stability.nodes.preprocess import ImgAuxMap
from stability.nodes.prompt import Prompt
from stability.nodes.txt2img import Txt2Img

ROOT = Path(__file__).resolve().parents[1]  # dags/ -> root
CONF = ROOT / 'node_conf'

with DAG('elven_worrior', out_dir=ROOT / 'outputs' / 'elven_warrior') as dag:
    components1 = FileImage(
        id='style1_img',
        path='~/images/woman_armor.png'
    )
    face = FaceIdEmbedImage(id='face_img', path='~/images/elven_woman.png')
    source = FileImage(id='source_img', path='~/images/warrior_anime.png')

    prompt = Prompt(
        id='elven_warrior',
        spec=CONF / 'jobs' / 'elven_warrior_prompt.conf'
    )

    sketch = ImgAuxMap(id='map', spec={'processor': 'canny'})

    img1 = Txt2Img(
        id='img1',
        spec=CONF / 'jobs' / 'elven_warrior_txt2img.conf'
    )

    prompt >> img1.prompt()

    source >> sketch >> img1.controlnet.add(
        'diffusers/controlnet-canny-sdxl-1.0',
        conditioning_scale=0.8,
        key='canny',
    )

    components1 >> img1.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter_sdxl_vit-h.bin',
        scale=0.7,
        key='style'
    )

    out = Img2Img(
        id='out',
        spec=CONF / 'jobs' / 'elf_princess_garden_i2i.conf'
    )

    face >> out.face_id.add(
        model_id="h94/IP-Adapter-FaceID",
        # ip-adapter-faceid_sdxl.bin brings style
        # ip-adapter-faceid-plusv2_sdxl.bin seems to better capture
        #    facial structure preserving more of  the original
        #    image style.
        weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
        scale=0.7,
        key="id",
    )

    img1 >> out
