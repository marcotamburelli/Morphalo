import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import torch

from morphalo.core.paths import make_node_output_path
from morphalo.dag import NodeRef
from morphalo.nodes.common.config_resolve import SpecInput, resolve_spec
from morphalo.nodes.ltx.preprocess.utils.depth_extractor import \
    DepthVideoExtractor

_model_id: str = 'Intel/dpt-hybrid-midas'
_device: str = 'cuda'
_autocast: bool = True


@dataclass
class VideoDepthMap(NodeRef):
    """
    Video depth preprocessing node producing a single packed control video.

    See DepthMap for the single-image analogue. This node processes an input video
    and writes a packed depth control video (mp4) plus a JSON sidecar.
    """

    spec: SpecInput = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

    def run(self, output_dir, input: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        input = input or {}
        upstream = input.get('default')
        if upstream is None:
            raise ValueError(
                'VideoDepthMap requires an input video wired to the default input.')

        in_path = upstream.get('video') or upstream.get('path')
        if not in_path:
            raise ValueError('Upstream output does not contain a video path.')

        spec = resolve_spec(self.spec)

        preserve_bg = bool(
            spec.get('preserve_bg', False)
        )  # usually False for depth
        max_frames = int(spec.get('max_frames', -1))

        # --- sizes ---
        # output/control resize (post-render). Kept as 'long_side' for backward compat.
        image_resolution = spec.get(
            'image_resolution',
            spec.get('long_side', 384)
        )
        # inference resize (pre-infer). Kept as 'long_side_infer' for backward compat.
        detect_resolution = spec.get(
            'detect_resolution',
            spec.get('long_side_infer', None)
        )

        # per_frame|global|running
        normalize = spec.get('normalize', 'running')
        invert = bool(spec.get('invert_depth', False))
        clip_p_low = float(spec.get('clip_p_low', 2.0))
        clip_p_high = float(spec.get('clip_p_high', 98.0))

        # temporal smoothing on normalized depth01 (EMA)
        temporal_ema_alpha = spec.get('temporal_ema_alpha', None)
        if temporal_ema_alpha is not None:
            temporal_ema_alpha = float(temporal_ema_alpha)

        # running global min/max (EMA over per-frame percentiles)
        running_minmax_alpha = float(spec.get('running_minmax_alpha', 0.9))

        fps_override = spec.get('fps', None)

        model = spec.get('model', {
            'id': _model_id,
            'device': _device,
            'autocast': _autocast,
        })

        out_dir = Path(output_dir)

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f'Unable to open video: {in_path}')

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 0.0
        in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        in_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        fps_out = float(fps_override) if fps_override else (
            fps_in if fps_in > 0 else 24.0)

        # build extractor (model cached internally via ModelCache)
        extractor = DepthVideoExtractor(
            model_id=model['id'],
            device=model['device'],
            autocast=model['autocast'],
        )

        # probe first frame to define output size
        ok, frame0 = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError('Empty video or failed first frame read.')

        # compute output frame (depth) size (post-render size)
        depth0 = extractor.render_frame(
            frame0,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            normalize='per_frame',
            invert=invert,
            clip_p_low=clip_p_low,
            clip_p_high=clip_p_high,
            temporal_ema_alpha=None  # probe only; don't warm EMA state
        )

        # If you want preserve_bg, you can overlay depth on original. Typically false.
        if preserve_bg:
            # simple overlay: use depth as luminance mask (debug)
            d0 = cv2.resize(
                depth0, (frame0.shape[1], frame0.shape[0]), interpolation=cv2.INTER_LINEAR)
            out0 = cv2.addWeighted(frame0, 0.6, d0, 0.4, 0.0)
        else:
            out0 = depth0

        out_h, out_w = out0.shape[:2]

        out_video_path = make_node_output_path(
            out_dir=out_dir,
            node_id=self.id,
            ext='mp4'
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(out_video_path), fourcc, fps_out, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f'Unable to open VideoWriter: {out_video_path}')

        # rewind
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # reset EMA state at clip start (important if node reused)
        extractor.reset_state()

        # optional global normalization pass (two-pass)
        # NOTE: global requires computing min/max over frames. Keeping it simple:
        global_minmax = None
        if normalize == 'global':
            # Pass 1: compute a global robust range from per-frame quantiles on raw depth.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ql = float(clip_p_low) / 100.0
            qh = float(clip_p_high) / 100.0

            global_lo = None
            global_hi = None
            idx = 0

            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                idx += 1
                if max_frames > 0 and idx > max_frames:
                    break

                d = extractor.predict_depth_tensor(
                    fr,
                    long_side_infer=detect_resolution
                )
                lo = torch.quantile(d, ql).item()
                hi = torch.quantile(d, qh).item()

                global_lo = lo if global_lo is None else min(global_lo, lo)
                global_hi = hi if global_hi is None else max(global_hi, hi)

            if global_lo is None or global_hi is None or global_hi <= global_lo:
                # Safety fallback
                normalize = 'per_frame'
                global_minmax = None
            else:
                global_minmax = (float(global_lo), float(global_hi))

            # Rewind for Pass 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            extractor.reset_state()

        # normalize='running' does NOT require a two-pass scan (EMA stats are built online).

        # main pass
        frames_total = 0
        frames_written = 0
        t0 = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_total += 1
            if max_frames > 0 and frames_total > max_frames:
                break

            depth_bgr = extractor.render_frame(
                frame,
                detect_resolution=detect_resolution,
                image_resolution=image_resolution,
                normalize=normalize,
                invert=invert,
                clip_p_low=clip_p_low,
                clip_p_high=clip_p_high,
                global_minmax=global_minmax,
            )

            if preserve_bg:
                d = cv2.resize(
                    depth_bgr, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                out = cv2.addWeighted(frame, 0.6, d, 0.4, 0.0)
            else:
                out = depth_bgr

            writer.write(out)
            frames_written += 1

        cap.release()
        writer.release()

        dt_s = time.perf_counter() - t0

        meta = {
            'ok': True,
            'node': self.op,
            'id': self.id,
            'video': str(out_video_path),
            'input': {
                'video': str(in_path),
                'fps': fps_in,
                'width': in_w,
                'height': in_h,
                'num_frames': in_n,
            },
            'params': {
                'preserve_bg': preserve_bg,
                'max_frames': max_frames,
                'image_resolution': image_resolution,
                'detect_resolution': detect_resolution,
                'normalize': normalize,
                'invert_depth': invert,
                'clip_p_low': clip_p_low,
                'clip_p_high': clip_p_high,
                'temporal_ema_alpha': temporal_ema_alpha,
                'running_minmax_alpha': running_minmax_alpha,
                'fps_out': fps_out,
                'fourcc': 'mp4v',
            },
            'stats': {
                'frames_total': frames_total,
                'frames_written': frames_written,
            },
            'model': model,
            'timing': {'seconds': round(dt_s, 3)}
        }

        meta_path = out_video_path.with_suffix('.json')
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

        return meta
