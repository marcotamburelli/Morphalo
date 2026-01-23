from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor

from stability.cache import CacheKey, ModelCache
from stability.nodes.ltx.preprocess.utils.drawing_utils import resize_long_side


@dataclass
class DepthVideoExtractor:
    """
    Depth estimator for video frames using DPT (MiDaS/DPT via HuggingFace).

    This extractor is responsible for:
    - lazy-loading and caching the processor/model
    - running per-frame depth estimation
    - converting depth to a 3-channel BGR control frame
    """

    model_id: str = "Intel/dpt-hybrid-midas"
    device: str = "cuda"
    autocast: bool = True

    # cached per-process handles (avoid repeated lookups in hot loop)
    _processor: Optional[DPTImageProcessor] = field(
        default=None, init=False, repr=False)
    _model: Optional[DPTForDepthEstimation] = field(
        default=None, init=False, repr=False)

    # --- optional temporal smoothing state (normalized depth in [0,1]) ---
    _ema_depth01: Optional[torch.Tensor] = field(
        default=None,
        init=False,
        repr=False
    )

    # --- optional "running" global normalization state (robust-ish) ---
    # We keep EMA of the chosen (low, high) quantiles to stabilize normalization.
    _ema_lo: Optional[float] = field(default=None, init=False, repr=False)
    _ema_hi: Optional[float] = field(default=None, init=False, repr=False)

    def _lazy_load(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        # Use your existing cache logic (same as get_depth_estimator)
        proc_key = CacheKey(kind="depth_processor",
                            ref=self.model_id, device="cpu", dtype="na")
        mod_key = CacheKey(kind="depth_model", ref=self.model_id,
                           device=self.device, dtype="na")

        processor = ModelCache.get(proc_key)
        if processor is None:
            processor = ModelCache.put(
                proc_key, DPTImageProcessor.from_pretrained(self.model_id))

        model = ModelCache.get(mod_key)
        if model is None:
            model = DPTForDepthEstimation.from_pretrained(
                self.model_id).to(self.device)
            model.eval()
            ModelCache.put(mod_key, model)

        self._processor = processor
        self._model = model

    @torch.no_grad()
    def predict_depth_tensor(
        self,
        frame_bgr: cv2.typing.MatLike,
        *,
        long_side_infer: int | None = None,
    ) -> torch.Tensor:
        """
        Return raw predicted depth as a (H, W) float tensor on the model device.
        """
        from PIL import Image

        self._lazy_load()
        assert self._processor is not None
        assert self._model is not None

        frame = np.asarray(frame_bgr)
        if long_side_infer is not None:
            frame = resize_long_side(frame, int(long_side_infer))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        pixel_values = self._processor(
            images=pil, return_tensors="pt").pixel_values.to(self.device)

        use_autocast = bool(self.autocast) and \
            str(self.device).startswith("cuda")

        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                depth = self._model(pixel_values).predicted_depth
        else:
            depth = self._model(pixel_values).predicted_depth

        d = depth[0]
        if d.ndim == 3:
            d = d[0]
        return d.to(dtype=torch.float32)

        # if use_autocast:
        #     with torch.autocast("cuda"):
        #         depth = self._model(pixel_values).predicted_depth
        # else:
        #     depth = self._model(pixel_values).predicted_depth

        # return depth[0].to(dtype=torch.float32)

    def reset_state(self) -> None:
        """
        Reset temporal smoothing state.

        Call this at the beginning of a new clip/video to avoid "bleeding"
        smoothing across unrelated sequences.
        """
        self._ema_depth01 = None
        self._ema_lo = None
        self._ema_hi = None

    def _update_running_minmax(self, lo: float, hi: float, alpha: float) -> Tuple[float, float]:
        if self._ema_lo is None or self._ema_hi is None:
            self._ema_lo, self._ema_hi = float(lo), float(hi)
        else:
            a = float(alpha)
            self._ema_lo = a * self._ema_lo + (1.0 - a) * float(lo)
            self._ema_hi = a * self._ema_hi + (1.0 - a) * float(hi)
        return float(self._ema_lo), float(self._ema_hi)

    @torch.no_grad()
    def render_frame(
        self,
        frame_bgr: cv2.typing.MatLike,
        *,
        detect_resolution: int | None = None,
        image_resolution: int | None = None,
        long_side_infer: int | None = None,  # legacy alias for detect_resolution
        normalize: Literal["per_frame", "global", "running"] = "per_frame",
        invert: bool = False,
        clip_p_low: float = 2.0,
        clip_p_high: float = 98.0,
        temporal_ema_alpha: Optional[float] = None,
        global_minmax: Optional[Tuple[float, float]] = None,
        running_minmax_alpha: float = 0.9
    ) -> cv2.typing.MatLike:
        """

        Estimate depth for a single frame and return a 3-channel BGR depth map.

        Parameters
        ----------
        frame_bgr : cv2.typing.MatLike
            Input frame in BGR format.
        detect_resolution : int or None
            Optional resize (long side) applied *before* inference (speed/quality tradeoff).
            If set, overrides ``long_side_infer``.
        image_resolution : int or None
            Optional resize (long side) applied *after* rendering the depth map,
            useful to match control-video resolution expectations.
        long_side_infer : int or None
            Legacy alias for ``detect_resolution`` (kept for compatibility).
        normalize : {"per_frame", "global", "running"}
            Normalization strategy for mapping depth to [0, 255].
            - ``per_frame``: robust per-frame scaling using percentiles.
            - ``global``: use explicit ``global_minmax=(min,max)``.
            - ``running``: EMA-stabilized min/max derived from per-frame percentiles.
        invert : bool
            If True, invert depth visualization (near/far swap).
        clip_p_low, clip_p_high : float
            Percentile clipping to reduce outlier influence before normalization.
        temporal_ema_alpha : float or None
            If provided (e.g. 0.7..0.9), apply EMA smoothing on the normalized
            depth map in [0,1]. If None, no temporal smoothing is applied.
        global_minmax : (float, float) or None
            If normalize="global", provide (min, max) depth values used for normalization.
        running_minmax_alpha : float
            EMA alpha for the running min/max when normalize="running".
            Higher values = more stable, slower adaptation. Default 0.9.

        Returns
        -------
        cv2.typing.MatLike
            Depth visualization as BGR uint8 (H, W, 3).
        """
        # Backwards-compatible input: if detect_resolution not provided, use legacy long_side_infer.
        if detect_resolution is None:
            detect_resolution = long_side_infer

        d = self.predict_depth_tensor(
            frame_bgr,
            long_side_infer=detect_resolution
        )

        if normalize == "global" and global_minmax is not None:
            glo, ghi = global_minmax
            lo = torch.tensor(glo, device=d.device, dtype=d.dtype)
            hi = torch.tensor(ghi, device=d.device, dtype=d.dtype)

            # scale (no hard clamp)
            d = (d - lo) / (hi - lo + 1e-8)

            # soft bound to [0, 1]
            d = torch.clamp(d, 0.0, 1.0)

        else:
            # per-frame robust scaling basis (percentile lo/hi)
            ql = float(clip_p_low) / 100.0
            qh = float(clip_p_high) / 100.0

            dq = d.float()  # quantile requires float32
            lo_t = torch.quantile(dq, ql)
            hi_t = torch.quantile(dq, qh)

            lo_v = float(lo_t.detach().cpu().item())
            hi_v = float(hi_t.detach().cpu().item())

            # Optional running global minmax (EMA of percentiles)
            if normalize == "running":
                lo_v, hi_v = self._update_running_minmax(
                    lo_v, hi_v, alpha=float(running_minmax_alpha)
                )

            lo = torch.tensor(lo_v, device=d.device, dtype=d.dtype)
            hi = torch.tensor(hi_v, device=d.device, dtype=d.dtype)

            # scale instead of clamp
            d = (d - lo) / (hi - lo + 1e-8)

            # soft bound
            d = torch.clamp(d, 0.0, 1.0)

        if invert:
            d = 1.0 - d

        # Optional temporal EMA smoothing on normalized depth in [0,1].
        # Default is disabled (temporal_ema_alpha=None) to preserve legacy behavior.
        if temporal_ema_alpha is not None:
            a = float(temporal_ema_alpha)
            # Keep it sane; don't hard fail in hot loop
            if a < 0.0:
                a = 0.0
            elif a > 1.0:
                a = 1.0

            if self._ema_depth01 is None or self._ema_depth01.shape != d.shape:
                self._ema_depth01 = d
            else:
                self._ema_depth01 = a * self._ema_depth01 + (1.0 - a) * d
            d = self._ema_depth01

        img = (d * 255.0).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Optional post-resize for output/conditioning
        if image_resolution is not None:
            out = resize_long_side(out, int(image_resolution))

        return out
