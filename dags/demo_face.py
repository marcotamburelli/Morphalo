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

with DAG('elven_warrior', out_dir=ROOT / 'outputs' / 'elven_warrior') as dag:
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
        spec=[
            CONF / 'jobs' / 'elf_princess_garden_i2i.conf',
            {'params': {'cfg': 5, 'strength': 0.45}}
        ]
    )

    prompt >> out.prompt()
    face >> out.face_id.add(
        model_id="h94/IP-Adapter-FaceID",
        # ip-adapter-faceid_sdxl.bin brings style
        # ip-adapter-faceid-plusv2_sdxl.bin seems to better capture
        #    facial structure preserving more of  the original
        #    image style.
        weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
        scale=0.5,
        key="id",
    )

    img1 >> out

with DAG('2_women', out_dir=ROOT / 'outputs' / '2_women') as dag:
    source = FileImage(
        id='source',
        path='~/images/2_woman.png'
    )
    face_1 = FaceIdEmbedImage(id='face_1', path='~/images/elven_woman.png')
    face_2 = FaceIdEmbedImage(id='face_2', path='~/images/mage_woman.png')

    styles = FileImage(id='styles', path=[
        '~/images/styles/female_armor/mix.png',
        '~/images/styles/female_gandalf/mix.png',
        '~/images/styles/garden/03.png'
    ])

    mask = FileImage(id='mask_fg', path=[
        '~/images/woman_left.png',  # subject 1
        '~/images/woman_right.png',  # subject 2
        '~/images/woman_left_right_bg.png'  # subject 2
    ])
    mask_1 = FileImage(id='mask_1', path='~/images/woman_left.png')
    mask_2 = FileImage(id='mask_2', path='~/images/woman_right.png')

    prompt = Prompt(
        id='2_women',
        spec=CONF / 'jobs' / '2_women_prompt.conf'
    )

    img1 = Img2Img(
        id='img1',
        spec=[
            CONF / 'jobs' / 'elf_princess_garden_i2i.conf',
            {'params': {'cfg': 8, 'strength': 0.7}}
        ]
    )

    source >> img1

    prompt >> img1.prompt()
    adapter = img1.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter_sdxl.bin',
        scale=[0.8, 0.8, 0.8],
        key='style'
    )
    styles >> adapter
    mask >> adapter.mask()

    # Attempt to properly apply faces
    out = Img2Img(
        id='out',
        spec=[
            CONF / 'jobs' / 'elf_princess_garden_i2i.conf',
            {'params': {'cfg': 5, 'strength': 0.4}}
        ]
    )

    prompt >> out.prompt()
    img1 >> out

    face_id_1 = out.face_id.add(
        model_id="h94/IP-Adapter-FaceID",
        weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
        scale=0.5,
        # clip_strength=0.5,
        key="id_1",
    )
    face_id_2 = out.face_id.add(
        model_id="h94/IP-Adapter-FaceID",
        weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
        scale=0.5,
        # clip_strength=0.5,
        key="id_2",
    )

    face_1 >> face_id_1
    face_2 >> face_id_2

    mask_1 >> face_id_1.mask()
    mask_2 >> face_id_2.mask()
