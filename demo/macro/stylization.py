from stability.dag import NodeGroup
from stability.nodes import Tap
from stability.nodes.common.config_resolve import SpecInput
from stability.nodes.img2img import Img2Img
from stability.nodes.preprocess import SubjectCrop


def stylize_subject_background_singlepass_group(
    name: str,
    *,
    img2img_spec: SpecInput,
    bg_adapter_scale: float,
    subject_adapter_scale: float,
    subject_mask_dilate_radius: int = 10,
    subject_mask_close_radius: int = 10,
    subject_mask_smoothing_radius: int = 20,
    bg_mask_dilate_radius: int = 10,
    bg_mask_close_radius: int = 10,
    bg_mask_smoothing_radius: int = 20,
) -> NodeGroup:
    """
    Single-pass subject/background stylization using two masked IP-Adapters.

    The group computes:
    - subject mask via SAM (mode='mask')
    - background mask via SAM (mode='negative-mask')

    Then runs a single Img2Img where:
    - background style adapter is masked by background mask
    - subject style adapter is masked by subject mask

    Ports
    -----
    - 'in_image' (Tap)
    - 'in_prompt' (Tap)
    - 'bg_style' (Tap)
    - 'subject_style' (Tap)

    Output
    ------
    The group output is the internal Img2Img node named 'out'.
    """

    with NodeGroup(name) as g:
        # -------------------
        # Ports
        # -------------------
        tap_image = Tap(name='in_image')
        tap_prompt = Tap(name='in_prompt')
        tap_bg_style = Tap(name='bg_style')
        tap_subject_style = Tap(name='subject_style')

        # -------------------
        # Masks (from input image)
        # -------------------
        subject_mask = SubjectCrop(
            name='subject_mask',
            spec={
                'model': {
                    'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                },
                'params': {
                    'mode': 'mask',
                    'dilate_radius': subject_mask_dilate_radius,
                    'close_radius': subject_mask_close_radius,
                    'smoothing_radius': subject_mask_smoothing_radius,
                },
            },
        )

        bg_mask = SubjectCrop(
            name='bg_mask',
            spec={
                'model': {
                    'sam_checkpoint': '~/models/sam/sam_vit_l_0b3195.pth',
                },
                'params': {
                    'mode': 'negative-mask',
                    'dilate_radius': bg_mask_dilate_radius,
                    'close_radius': bg_mask_close_radius,
                    'smoothing_radius': bg_mask_smoothing_radius,
                },
            },
        )

        # -------------------
        # Single Img2Img pass
        # -------------------
        out = Img2Img(
            name='out',
            spec=img2img_spec,
        )

        bg_adapter = out.ip_adapter.add(
            'h94/IP-Adapter',
            subfolder='sdxl_models',
            weight_name='ip-adapter_sdxl_vit-h.bin',
            scale=bg_adapter_scale,
            key='bg_style',
        )

        subject_adapter = out.ip_adapter.add(
            'h94/IP-Adapter',
            subfolder='sdxl_models',
            weight_name='ip-adapter_sdxl_vit-h.bin',
            scale=subject_adapter_scale,
            key='subject_style',
        )

        # -------------------
        # Wiring
        # -------------------
        tap_image >> subject_mask
        tap_image >> bg_mask

        tap_image >> out
        tap_prompt >> out.prompt()

        tap_bg_style >> bg_adapter
        tap_subject_style >> subject_adapter

        bg_mask >> bg_adapter.mask()
        subject_mask >> subject_adapter.mask()

        # -------------------
        # Register ports
        # -------------------
        g.register_ports(tap_image, tap_prompt,
                         tap_bg_style, tap_subject_style)

    return g
