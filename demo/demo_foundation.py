from pathlib import Path

from stability.dag import DAG
from stability.nodes.file_image import FileImage
from stability.nodes.foundation import QwenImage, QwenImageEdit
from stability.nodes.prompt import Prompt

ROOT = Path(__file__).resolve().parents[1]  # dags/ -> root
CONF = ROOT / 'node_conf'

with DAG('foundation', out_dir=ROOT / 'outputs' / 'foundation') as dag:
    prompt = Prompt(
        id='prompt',
        spec=CONF / 'jobs' / 'foundation_prompt.conf',
    )

    out = QwenImage(
        id='out',
        spec={
            'params': {
                'steps': 40,
                'true_cfg_scale': 4,
                'width': 1024,
                'height': 1024,
            }
        }
    )

    prompt >> out.prompt()

with DAG('foundation', out_dir=ROOT / 'outputs' / 'foundation') as dag:
    prompt = Prompt(
        id='prompt',
        spec=CONF / 'jobs' / 'foundation_prompt.conf',
    )

    out = QwenImage(
        id='out',
        spec={
            'params': {
                'steps': 40,
                'true_cfg_scale': 4,
                'width': 1024,
                'height': 1024,
            }
        }
    )

    prompt >> out.prompt()

with DAG('elven_woman', out_dir=ROOT / 'outputs' / 'elven_woman') as dag:
    face = FileImage(id='face_img', path='~/images/elven_woman.png')

    prompt = Prompt(
        id='prompt',
        spec={
            'lang': 'ita_Latn',
            'prompt': [
                "Converti l'immagine in una fotografia.",
                "Ragazza elfica sulla ventina, pelle fresca con orecchie a punta.",
                "Vestito estivo: maglietta leggera.",
                "Mantieni la stessa identità: contorni, tratti del viso, distanza tra occhi, forma di naso, labbra e orecchie (importante mantenere la forma e dimensione delle orecchie), colore degli occhi, forma e dimensioni del mento, carnagione e capelli.",
                "Le orecchie devono essere a punta da elfo.",
                "Illuminazione morbida e diffusa da davanti (softbox), esposizione corretta.",
                "Pelle liscia e naturale con pori e micro-dettagli."
            ],
            'negative_prompt': [
                "rughe",
                "trucco, mascara, ombretto",
                # "effetto beauty",
                # "smoothing",
                "bassa qualità",
                "sfocato",
                "rumore",
                "pelle di plastica",
                "pelle da bambola",
                "effetto cera",
                "occhi storti",
                "occhi asimmetrici",
                "stile cartoon",
                "stile manga",
                "render 3D",
                "CGI"
            ],
        },
    )

    out = QwenImageEdit(
        id='out',
        spec={
            'params': {
                'steps': 50,
                'true_cfg_scale': 4,
            }
        }
    )

    face >> out
    prompt >> out.prompt()


with DAG('mage_woman', out_dir=ROOT / 'outputs' / 'mage_woman') as dag:
    face = FileImage(id='face_img', path='~/images/mage_woman.png')

    prompt = Prompt(
        id='prompt',
        spec={
            'lang': 'ita_Latn',
            'prompt': [
                "Converti l'immagine in una fotografia frontale (headshot).",
                "Ragazza sulla ventina.",
                "Vestito estivo: maglietta leggera.",
                "Mantieni la stessa identità: contorni, tratti del viso, distanza tra occhi, forma di naso, labbra e orecchie, forma e dimensioni del mento, carnagione e capelli.",
                "Carnagione eburnea, capelli color argento raccolti a cipolla, prondi occhi viola.",
                "Occhi profondi di colore viola intenso."
                "Illuminazione morbida e diffusa da davanti (softbox), esposizione corretta.",
                "Pelle liscia naturale con pori e micro-dettagli."
            ],
            'negative_prompt': [
                "rughe",
                "trucco, mascara, ombretto",
                # "effetto beauty",
                # "smoothing",
                "bassa qualità",
                "sfocato",
                "rumore",
                "pelle di plastica",
                "pelle da bambola",
                "effetto cera",
                "occhi storti",
                "occhi asimmetrici",
                "stile cartoon",
                "stile anime",
                "render 3D",
                "CGI"
            ],
        },
    )

    out = QwenImageEdit(
        id='out',
        spec={
            'params': {
                'steps': 40,
                'true_cfg_scale': 4,
            }
        }
    )

    face >> out
    prompt >> out.prompt()
