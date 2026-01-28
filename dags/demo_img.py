from pathlib import Path

from stability.dag import DAG
from stability.nodes.file_image import FileImage
from stability.nodes.img2img import Img2Img
from stability.nodes.preprocess import ImgAuxMap
from stability.nodes.prompt import Prompt
from stability.nodes.txt2img import Txt2Img

ROOT = Path(__file__).resolve().parents[1]  # dags/ -> root
CONF = ROOT / 'node_conf'

with DAG('elven_worrior', out_dir=ROOT / 'outputs' / 'elven_warrior') as dag:
    components = FileImage(id='style_img', path=[
        *[f'~/images/styles/female_armor/{i:02d}.png' for i in range(1, 9)],
        *[f'~/images/styles/garden/{i:02d}.png' for i in range(1, 9)],
    ])
    face = FileImage(id='face_img', path='~/images/elven_woman.png')
    source = FileImage(id='canny_img', path='~/images/warrior_anime.png')

    prompt = Prompt(
        id='elven_warrior',
        spec=CONF / 'jobs' / 'elven_warrior_prompt.conf'
    )

    sketch = ImgAuxMap(id='map', spec={'processor': 'canny'})

    out = Txt2Img(
        id='gen',
        spec=CONF / 'jobs' / 'elven_warrior_txt2img.conf'
    )

    prompt >> out.prompt()

    source >> sketch >> out.controlnet.add(
        'diffusers/controlnet-canny-sdxl-1.0',
        conditioning_scale=0.4,
        key='canny',
    )

    face >> out.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        # weight_name='ip-adapter-plus-face_sdxl_vit-h.safetensors',
        weight_name='ip-adapter-plus-face_sdxl_vit-h.safetensors',
        scale=0.3,
        key='face'
    )

    components >> out.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter_sdxl_vit-h.bin',
        scale=0.7,
        key='style'
    )

with DAG('elven_worrior_2', out_dir=ROOT / 'outputs' / 'elven_warrior_2') as dag:
    # --- reference images ---
    refs = FileImage(
        id='armor_refs',
        path=[
            f'~/images/styles/female_armor/05.png',
            f'~/images/styles/garden/08.png'
        ]
    )

    # subject / face / source
    face = FileImage(id='face_img', path='~/images/elven_woman.png')
    source = FileImage(id='model_img', path='~/images/warrior_anime.png')

    # masks: white=subject on black background
    # background mask should be the inverse (white=background)
    mask = FileImage(id='mask_fg', path=[
        '~/images/mask.png',  # subject
        '~/images/mask_bg.png'  # background
    ])

    # prompt
    prompt = Prompt(
        id='elven_warrior',
        spec=CONF / 'jobs' / 'elven_warrior_prompt.conf',
    )

    # aux map for T2I-Adapter (sketch/softedge)
    sketch = ImgAuxMap(
        id='sketch',
        spec={
            'processor': 'lineart_realistic',
            'detect_long_side': 1024,
        },
    )

    # helper txt2img (to produce a good init image)
    help_img = Txt2Img(
        id='help_img',
        spec=CONF / 'jobs' / 'elven_warrior_txt2img.conf',
    )

    prompt >> help_img.prompt()

    source >> sketch >> help_img.t2i_adapter.add(
        'TencentARC/t2i-adapter-lineart-sdxl-1.0',
        conditioning_scale=0.5,
        key='sketch',
    )

    # final img2img
    out = Img2Img(
        id='out',
        spec=CONF / 'jobs' / 'elf_princess_garden_i2i.conf',
    )

    prompt >> out.prompt()
    help_img >> out

    # --- IP-Adapter: face (global or add face-only mask if you want) ---
    face >> out.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter-plus-face_sdxl_vit-h.safetensors',
        scale=0.2,
        key='face_2',
    )

    # --- IP-Adapter: style
    ip_style = out.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter_sdxl_vit-h.bin',
        scale=0.7,
        key="style",
    )
    refs >> ip_style
    mask >> ip_style.mask()
