from pathlib import Path

from stability.dag import DAG
from stability.nodes import FileImage, Img2Img, Prompt, Txt2Img
from stability.nodes.face_id_embed_image import FaceIdEmbedImage
from stability.nodes.preprocess import ImageStack, SubjectCrop

ROOT = Path(__file__).resolve().parents[1]  # dags/ -> root
CONF = ROOT / 'node_conf'


with DAG('crop', out_dir=ROOT / 'outputs' / 'crop') as dag:
    SubjectCrop(path='~/images/elven_princess.png', spec={
        'model': {
            'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth'
        },
    })

with DAG('crop_head', out_dir=ROOT / 'outputs' / 'crop_head') as dag:
    background = FileImage(id='bk', path='~/images/styles/space/01.png')

    crop = SubjectCrop(
        id='crop',
        path='~/images/keyframe.png',
        spec={
            'model': {
                'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
            },
            'params': {
                'target': 'head',
                'expansion': 1.5,
                'box_margin': 0.0,
                'full_frame': False
            },
        },
    )

    stack = ImageStack(id='stack', spec={
        'params': {}
    })

    background >> stack.image(0)
    crop >> stack.image(
        1,
        feather=30,
        position='bottom-right',
        resize=(600, None)
    )


with DAG('extract_mask', out_dir=ROOT / 'outputs' / 'mask') as dag:
    SubjectCrop(path='~/images/elven_princess.png', spec={
        'model': {
            'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
            'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
        },
        'params': {
            'target': 'head',
            'mode': 'mask',
            'dilate_radius': 10,
            'close_radius': 10,
            'smoothing_radius': 40,
            'expansion': 2,
        }
    })

with DAG('stack', out_dir=ROOT / 'outputs' / 'stack') as dag:
    background = FileImage(id='bk', path='~/images/styles/space/01.png')

    prompt = Prompt(id='prompt', spec={
        'lang': 'ita_Latn',
        'prompt': {
            'content': [
                "Ragazza, capelli color argento, riccioli, lunghi, occhi azzurri.",
                "Immagine lontana dalla telecamera.",
                "Cammina in un deserto. Galassie e pianeti si vedono nel cielo notturno."
            ],
            'style': [
                "Ragazza giovane e bella, volto affilato con mento prominente.",
                "Capelli grigi come l'argento e splendenti, lunghi e riccioli, mossi dal vento. Occhi azzurri penetranti.",
                # "Pelle levigata e brillante.",
                "Alta, longilinea, Seno piccolo, coppa A.",
                "Carnagione pallida, chiara, pelle liscia.",
                "Stile Lord of The Rings, Studio Ghibli; tratti facciali Final Fantasy.",
                "Cammina in un deserto; il cielo notturno con galassie pianeti stile Hubble, deep scan.",
            ]
        },
        'negative_prompt': [
            "barba, abbronzatura",
            "bassa qualità",
            "sfocato",
            "rumore",
            "plastic skin",
            "doll skin",
            "effetto cera",
            "occhi storti",
            "occhi asimmetrici",
            "stile cartoon",
            "stile anime",
            "render 3D",
            "disegno"
        ]
    })

    img1 = Txt2Img(
        id='img1',
        spec=[
            CONF / 'common.conf',
            {'params': {'cfg': 5}}
        ]
    )

    prompt >> img1.prompt()

    crop = SubjectCrop(id='crop', spec={
        'model': {
            'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth'
        },
        'params': {
            'full_frame': True
        }
    })

    stack = ImageStack(id='stack', spec={
        'params': {}
    })

    background >> stack.image(0)
    img1 >> crop >> stack.image(1, feather=5)

    out = Img2Img(
        id='out',
        spec=[
            CONF / 'common.conf',
            {'params': {'cfg': 7, 'strength': 0.7}}
        ]
    )

    prompt >> out.prompt()
    stack >> out


with DAG('refine', out_dir=ROOT / 'outputs' / 'refine') as dag:
    img_source = FileImage(id='src', path='~/images/out/elven_warrior.png')
    face = FaceIdEmbedImage(id='face_img', path='~/images/out/GeT/yak.png')

    crop_face = SubjectCrop(
        id='crop_face',
        spec={
            'model': {
                'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                'face_landmarker_task': '~/models/mediapipe/face_landmarker.task',
            },
            'params': {
                'target': 'head',
                'mode': 'default',
                'crop_mode': 'bbox',
                'box_margin': 0.12,
                'expansion': 1.2,
            },
        }
    )

    refine_face = Img2Img(
        id='refine_face',
        spec=[
            CONF / 'common.conf',
            {
                'params': {'cfg': 5, 'strength': 0.4},

                'prompt': {
                    'content': [
                        "Elven warrior princess. Pointed ears.",
                        "Oval face with a prominent chin. Black hair, blue eyes.",
                    ],
                    'style': [
                        "Young. Natural skin. Pale, fair complexion. Black hair. Deep blue eyes. Long, pointed elf ears.",
                        "Lamellar armor, of polished black steel, with intricate fantasy-style gold inlays.",
                        "Photography; Lord of the Rings, Studio Ghibli style; Final Fantasy facial features.",
                    ]
                },

                'negative_prompt': [
                    "out of focus",
                    "wrinkles",
                    "makeup, mascara, eyeshadow",
                    "smoothing",
                    "low quality",
                    "blur",
                    "noise",
                    "plastic skin",
                    "doll skin",
                    "wax effect",
                    "crossed eyes",
                    "asymmetrical eyes",
                    "cartoon style",
                    "anime style",
                    "3D rendering",
                    "drawing"
                ]

            }
        ]
    )
    face_id = refine_face.face_id.add(
        model_id="h94/IP-Adapter-FaceID",
        # ip-adapter-faceid_sdxl.bin brings style
        # ip-adapter-faceid-plusv2_sdxl.bin seems to better capture
        #    facial structure preserving more of  the original
        #    image style.
        weight_name='ip-adapter-faceid-plusv2_sdxl.bin',
        scale=0.7,
        # clip_strength=0.1,
        key="id",
    )

    face >> face_id

    stack = ImageStack(
        id='merge',
        spec={
            'params': {
                'width': 1024,
                'height': 1024,
                'background': None,
                'out_mode': 'RGBA',
            }
        }
    )

    # -------------------
    # wiring
    # -------------------

    # 1) base image
    img_source >> stack.image(0)

    # 2) crop face from source
    img_source >> crop_face

    # 3) refine cropped face
    crop_face >> refine_face

    # 4) overlay refined face on top, using crop metadata (anchor + bbox size)
    layer1 = stack.image(1, position='center', resize=None, feather=30)

    refine_face >> layer1                # actual pixels to overlay
    crop_face >> layer1.transform()    # provides anchor_xy + bbox_size
