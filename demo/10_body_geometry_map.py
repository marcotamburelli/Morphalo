"""
MHR body reconstruction and proportion transfer demo.

This example renders two mannequins in the pose and camera frame of the target
image. The first uses the target's own proportions. The second uses the mean
proportions estimated from reference photos of another person.

Conceptual graph
----------------
Target image -----> BodyGeometryMap (target_body) -> RGBA PNG
Target image -----> BodyGeometryMap (transferred_body) -> RGBA PNG
Reference views --> transferred_body.source()

How to run
----------
Set the input image paths. From the repository root, run:

    ./bin/run_dag.sh demo.10_body_geometry_map --dag body_geometry_map

Inputs (local paths)
--------------------
- Target image: a photo of the person in the desired pose.
- Reference views: one photo, a list of photos, or a directory of photos of
  the same person. In each input, the node uses the largest detected person box.
- MHR rig: run ``./fetch_mhr.sh`` to install ``~/models/mhr/mhr_model.pt``,
  or point ``model.mhr_model_pt`` at an existing local copy.
- InstantHMR: the node downloads the Medium-commercial checkpoint from
  Hugging Face on first use and reuses the Hub cache.
- Person detector: Ultralytics loads ``yolov8n.pt`` from ``~/models/yolo``
  or downloads it to that directory on first use.
- Pose refinement: run ``./fetch_media_pipe.sh`` and provide the pose task.
  Both renders fit the pelvis and spine, refine projected torso widths, then
  align the limbs using MediaPipe joint positions and relative depth order.

Outputs
-------
The two RGBA PNGs are saved under ``outputs/body_geometry_map``. Both have a
transparent background by default. Set ``background_color`` to an RGB color
to render on an opaque background.
The debug folders contain a ``pose_overlay.png`` comparing the rendered
skeleton with the target landmarks.
"""

from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes import FileImage
from morphalo.nodes.preprocess import BodyGeometryMap

ROOT = Path(__file__).resolve().parents[1]

TARGET_IMG = '~/images/init_img_4.png'

SOURCE_VIEWS = '~/images/body_x'

BODY_SPEC = {
    'model': {
        'device': 'cuda',
        'mhr_model_pt': '~/models/mhr/mhr_model.pt',
        'pose_landmarker_task': '~/models/mediapipe/pose_landmarker_heavy.task',
    },
    'params': {
        'body_color': '#b8b8b8',
        'background_color': None,
    },
    'debug': {'save_debug': True},
}

with DAG(
    name='body_geometry_map',
    out_dir=ROOT / 'outputs' / 'body_geometry_map',
):
    target = FileImage(name='target', path=TARGET_IMG)
    references = FileImage(name='references', path=SOURCE_VIEWS)

    target_body = BodyGeometryMap(
        name='target_body',
        spec=BODY_SPEC,
    )
    target >> target_body

    transferred_body = BodyGeometryMap(
        name='transferred_body',
        spec={
            **BODY_SPEC,
            'params': {
                'body_color': '#b8b8b8',
                'background_color': None,
                'gender': 'f',
                'chest_volume': 2,
                'hip_volume': 1,
            },
        },
    )
    target >> transferred_body
    references >> transferred_body.source()
