from pathlib import Path

from stability.dag import DAG
from stability.nodes import FaceIdEmbedImage, FileImage, Inpaint, Prompt
from stability.nodes.img2img import Img2Img

ROOT = Path(__file__).resolve().parents[1]  # dags/ -> root
CONF = ROOT / 'node_conf'

with DAG('elven_warrior', out_dir=ROOT / 'outputs' / 'elven_warrior') as dag:
    source = FileImage(id='source', path='~/images/keyframe.png')
    face_emb = FaceIdEmbedImage(id='face_img', path='~/images/elven_woman.png')
    mask = FileImage(id='face_mask', path='~/images/mask_face.png')

    prompt = Prompt(
        id='elven_warrior',
        spec=CONF / 'jobs' / 'elven_warrior_prompt_inpaint.conf'
    )

    out = Inpaint(
        id='out',
        spec=[
            CONF / 'jobs' / 'elven_warrior_txt2img.conf',
            {'params': {'cfg': 7, 'strength': 0.5}}
        ]
    )

    prompt >> out.prompt()
    mask >> out.mask()
    source >> out

    face_emb >> out.face_id.add(
        model_id="h94/IP-Adapter-FaceID",
        weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
        scale=0.7,
        clip_strength=0.5,
        key="id",
    )


with DAG('2_women', out_dir=ROOT / 'outputs' / '2_women_inpaint') as dag:
    source = FileImage(
        id='source',
        path='~/images/2_woman_old.png'
    )
    face_1 = FaceIdEmbedImage(id='face_1', path='~/images/elven_woman.png')
    face_2 = FaceIdEmbedImage(id='face_2', path='~/images/mage_woman.png')

    styles = FileImage(
        id='styles',
        path=[
            f'~/images/styles/female_armor/01.png',
            f'~/images/styles/female_gandalf/01.png',
            f'~/images/styles/garden/08.png',
        ]
    )
    # style_1 = FileImage(
    #     id='style_1',
    #     path='~/images/styles/female_armor/01.png'
    # )
    # style_2 = FileImage(
    #     id='style_2',
    #     path='~/images/styles/female_gandalf/01.png'
    # )
    mask_tmp = FileImage(
        id='mask_tmp',
        path=[
            '~/images/woman_left.png',
            '~/images/woman_right.png',
            '~/images/woman_left_right_bg.png'
        ]
    )

    mask_1 = FileImage(id='mask_1', path='~/images/woman_left.png')
    mask_2 = FileImage(id='mask_2', path='~/images/woman_right.png')

    prompt = Prompt(
        id='common_prompt',
        spec=CONF / 'jobs' / '2_women_prompt.conf'
    )
    prompt_1 = Prompt(
        id='warrior_prompt',
        spec=CONF / 'jobs' / 'elven_warrior_prompt_inpaint.conf'
    )
    prompt_2 = Prompt(
        id='mage_prompt',
        spec=CONF / 'jobs' / 'mage_prompt_inpaint.conf'
    )

    img_tmp = Img2Img(
        id='img_tmp',
        spec=[
            CONF / 'jobs' / 'elf_princess_garden_i2i.conf',
            {'params': {'cfg': 7, 'strength': 0.55}}
        ]
    )

    source >> img_tmp

    prompt >> img_tmp.prompt()
    adapter = img_tmp.ip_adapter.add(
        'h94/IP-Adapter',
        subfolder='sdxl_models',
        weight_name='ip-adapter_sdxl.bin',
        scale=0.6,
        key='style'
    )
    styles >> adapter
    mask_tmp >> adapter.mask()

    # Attempt to properly apply faces through inpaint

    # inpainting first face in the first masked area
    inpaint_style_1 = Inpaint(
        id='inpaint_style_1',
        spec=[
            CONF / 'jobs' / 'elven_warrior_txt2img.conf',
            {'params': {'cfg': 3.5, 'strength': 0.6}}
        ]
    )

    prompt_1 >> inpaint_style_1.prompt()
    mask_1 >> inpaint_style_1.mask()
    img_tmp >> inpaint_style_1

    # style_1 >> inpaint_style_1.ip_adapter.add(
    #     'h94/IP-Adapter',
    #     subfolder='sdxl_models',
    #     weight_name='ip-adapter_sdxl.bin',
    #     scale=0.4,
    #     key='style_1'
    # )

    inpaint_face_1 = Inpaint(
        id='inpaint_face_1',
        spec=[
            CONF / 'jobs' / 'elven_warrior_txt2img.conf',
            {
                'params': {'cfg': 3, 'strength': 0.4},
                'prompt': 'Elven warrior with metal plate armor, black hair, blue eyes, pointed elf ears.'
            }
        ]
    )

    mask_1 >> inpaint_face_1.mask()
    inpaint_style_1 >> inpaint_face_1

    face_1 >> inpaint_face_1.face_id.add(
        model_id="h94/IP-Adapter-FaceID",
        weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
        scale=0.5,
        # clip_strength=0.7,
        key="id",
    )

    # inpainting second face in the first masked area
    inpaint_style_2 = Inpaint(
        id='inpaint_style_2',
        spec=[
            CONF / 'jobs' / 'elven_warrior_txt2img.conf',
            {'params': {'cfg': 3.5, 'strength': 0.6}}
        ]
    )

    prompt_2 >> inpaint_style_2.prompt()
    mask_2 >> inpaint_style_2.mask()
    inpaint_face_1 >> inpaint_style_2

    # style_2 >> inpaint_style_2.ip_adapter.add(
    #     'h94/IP-Adapter',
    #     subfolder='sdxl_models',
    #     weight_name='ip-adapter_sdxl.bin',
    #     scale=0.5,
    #     key='style_1'
    # )

    inpaint_face_2 = Inpaint(
        id='out',
        spec=[
            CONF / 'jobs' / 'elven_warrior_txt2img.conf',
            {
                'params': {'cfg': 3, 'strength': 0.5},
                'prompt': 'Young woman, silver hair, deep violet eyes, white and delicate eyebrows'
            }
        ]
    )

    mask_2 >> inpaint_face_2.mask()
    inpaint_style_2 >> inpaint_face_2

    face_2 >> inpaint_face_2.face_id.add(
        model_id="h94/IP-Adapter-FaceID",
        weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
        scale=0.7,
        # clip_strength=0.7,
        key="id",
    )
