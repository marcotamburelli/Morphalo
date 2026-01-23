import cv2
import numpy as np

from stability.nodes.ltx.preprocess.utils.drawing_utils import resize_long_side


class CannyExtractor:
    """
    Canny edge extractor and renderer for video control maps.

    This implementation is "video-friendly" (extra knobs: blur/aperture/L2/dilate),
    and also supports the ControlNet-style two-stage sizing:
      - detect at ``detect_resolution`` (long side) for stability/speed
      - output at ``image_resolution`` (long side) for downstream alignment

    If ``image_resolution`` is not provided, ``long_side`` is used as a legacy
    output resize parameter.
    """

    def render_frame(
        self,
        frame_bgr: cv2.typing.MatLike,
        *,
        low_threshold: int = 80,
        high_threshold: int = 160,
        blur_ksize: int = 5,
        blur_sigma: float = 0.0,
        aperture_size: int = 3,
        l2_gradient: bool = False,
        dilate: int = 0,
        dilate_iter: int = 1,
        preserve_bg: bool = False,
        invert: bool = False,
        detect_resolution: int | None = None,
        image_resolution: int | None = None,
        long_side: int | None = None,
    ) -> cv2.typing.MatLike:
        """
        Render a Canny edge overlay for a single frame.

        Parameters
        ----------
        frame_bgr : cv2.typing.MatLike
            Input frame in BGR format.
        low_threshold, high_threshold : int
            Canny thresholds.
        blur_ksize : int
            Gaussian blur kernel size (odd). Use 0 or 1 to disable.
        blur_sigma : float
            Gaussian sigma. 0 lets OpenCV choose automatically.
        aperture_size : int
            Canny Sobel aperture size (3, 5, or 7).
        l2_gradient : bool
            Use a more precise L2 norm for gradient magnitude.
        dilate : int
            If >0, apply morphological dilation to thicken edges.
        dilate_iter : int
            Dilation iterations.
        preserve_bg : bool
            If True, overlay edges on the original frame; else black background.
        invert : bool
            If True, invert edge colors (useful for some pipelines).
        detect_resolution : int or None
            If provided, resize the input frame so its long side equals this value
            *before* computing edges. This often improves stability and performance.
        image_resolution : int or None
            If provided, resize the output map so its long side equals this value
            *after* computing edges / overlay, similar to ``controlnet_aux`` behavior.
        long_side : int or None
            Legacy output resize parameter (used only if ``image_resolution`` is None).

        Returns
        -------
        cv2.typing.MatLike
            Output BGR frame.
        """
        # Ensure we operate on a NumPy array (MatLike may be UMat, etc.)
        frame = np.asarray(frame_bgr)

        # --- Stage 1: detection resize (controlnet_aux-style) ---
        if detect_resolution is not None:
            frame_det = resize_long_side(frame, int(detect_resolution))
        else:
            frame_det = frame

        gray = cv2.cvtColor(frame_det, cv2.COLOR_BGR2GRAY)

        if blur_ksize and blur_ksize > 1:
            # OpenCV requires odd kernel sizes
            if blur_ksize % 2 == 0:
                blur_ksize += 1
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)

        # Canny aperture size must be 3, 5, or 7
        if aperture_size not in (3, 5, 7):
            raise ValueError(
                f"aperture_size must be 3, 5, or 7, got {aperture_size}"
            )

        edges = cv2.Canny(
            gray,
            threshold1=int(low_threshold),
            threshold2=int(high_threshold),
            apertureSize=int(aperture_size),
            L2gradient=bool(l2_gradient),
        )

        if dilate and dilate > 0:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=int(dilate_iter))

        if invert:
            edges = 255 - edges

        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if preserve_bg:
            # Overlay on the (possibly resized) detection frame
            out = frame_det.copy()
            m = edges > 0
            out[m] = (255, 255, 255)
        else:
            out = edges_bgr

        # --- Stage 2: output resize ---
        # Priority: image_resolution (new) > long_side (legacy) > no resize
        target = image_resolution if image_resolution is not None else long_side
        if target is not None:
            out = resize_long_side(out, int(target))

        return out
