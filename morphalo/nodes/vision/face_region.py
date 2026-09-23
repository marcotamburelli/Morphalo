import numpy as np

from morphalo.nodes.preprocess.utils.geometry import (offset_bbox_xyxy,
                                                      offset_landmarks_xy)


MEDIAPIPE_JAWLINE = (
    172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365, 397,
)


def mp_face_landmarks(
    img_rgb: np.ndarray,
    *,
    face_landmarker,
) -> np.ndarray:
    """
    Detect MediaPipe face landmarks and select the largest detected face.

    Every MediaPipe hypothesis is converted from normalized coordinates to
    clipped integer pixel coordinates. When multiple faces are present, the
    hypothesis with the largest axis-aligned landmark bbox is returned.

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB image with shape ``(height, width, 3)``.

    face_landmarker : object
        Initialized MediaPipe FaceLandmarker exposing ``detect``.

    Returns
    -------
    np.ndarray
        Integer landmark coordinates with shape ``(N, 2)`` in ``(x, y)``
        pixel order, clipped to the image bounds.

    Raises
    ------
    RuntimeError
        If MediaPipe detects no faces or every detected landmark set produces
        an invalid bbox.

    Notes
    -----
    Selection is based only on projected landmark area. The function does not
    perform identity tracking or associate the face with a particular person.
    """
    import mediapipe as mp

    h, w = img_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb,
    )

    result = face_landmarker.detect(mp_image)
    face_landmarks_list = result.face_landmarks or []

    if not face_landmarks_list:
        raise RuntimeError('MediaPipe found no face landmarks.')

    best_xy = None
    best_area = None

    for landmarks in face_landmarks_list:
        face_xy = np.empty((len(landmarks), 2), dtype=np.int32)

        for i, lm in enumerate(landmarks):
            x = int(round(float(lm.x) * w))
            y = int(round(float(lm.y) * h))
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            face_xy[i] = (x, y)

        x1 = np.min(face_xy[:, 0])
        y1 = np.min(face_xy[:, 1])
        x2 = np.max(face_xy[:, 0]) + 1
        y2 = np.max(face_xy[:, 1]) + 1

        if x2 <= x1 or y2 <= y1:
            continue

        area = float((x2 - x1) * (y2 - y1))

        if best_area is None or area > best_area:
            best_area = area
            best_xy = face_xy

    if best_xy is None:
        raise RuntimeError('Could not derive valid face landmarks.')

    return best_xy


def face_bbox_xyxy_from_landmarks(
    face_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    expansion: float = 1.0,
) -> tuple[int, int, int, int]:
    """
    Compute an end-exclusive face bounding box from face landmarks.

    Parameters
    ----------
    face_xy : np.ndarray
        Pixel-space face landmarks with shape ``(N, 2)`` in ``(x, y)`` order.

    image_shape : tuple[int, ...]
        Shape of the image containing the landmarks. The first two values are
        interpreted as ``(height, width)``.

    expansion : float, default=1.0
        Multiplicative width and height expansion around the bbox center.
        Values below one contract the box; zero produces the smallest valid
        one-pixel box after clipping.

    Returns
    -------
    tuple[int, int, int, int]
        Clipped end-exclusive bbox ``(x1, y1, x2, y2)``.

    Raises
    ------
    ValueError
        If ``expansion`` is negative.
    RuntimeError
        If ``face_xy`` is not an ``(N, 2)`` array or no valid bbox can be
        derived inside the image.
    """
    if face_xy.ndim != 2 or face_xy.shape[1] != 2:
        raise RuntimeError('Invalid face landmarks.')

    if expansion < 0:
        raise ValueError(
            f'Invalid expansion={expansion!r}; expected >= 0.'
        )

    h, w = image_shape[:2]

    x1 = int(np.min(face_xy[:, 0]))
    y1 = int(np.min(face_xy[:, 1]))
    x2 = int(np.max(face_xy[:, 0])) + 1
    y2 = int(np.max(face_xy[:, 1])) + 1

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError('Invalid face bbox.')

    if expansion != 1.0:
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        bw *= float(expansion)
        bh *= float(expansion)

        x1 = int(np.floor(cx - bw / 2.0))
        y1 = int(np.floor(cy - bh / 2.0))
        x2 = int(np.ceil(cx + bw / 2.0))
        y2 = int(np.ceil(cy + bh / 2.0))

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError('Invalid face bbox.')

    return x1, y1, x2, y2


def face_side_of_jaw_mask(
    face_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    support_mask: np.ndarray,
    margin_ratio: float = 0.03,
) -> np.ndarray:
    """
    Build the image region on the facial side of the MediaPipe jawline.

    The jaw arc from landmarks 172 through chin landmark 152 to landmark 397 is
    shifted slightly toward the neck to preserve the visible chin. Each jaw
    endpoint is connected to the nearest boundary point of ``support_mask``,
    producing a barrier across the semantic face-and-neck region. The returned
    side is the component containing nose landmark 1.

    Parameters
    ----------
    face_xy : np.ndarray
        Pixel-space MediaPipe face landmarks with shape ``(N, 2)``. The array
        must contain landmark indices through 397, including forehead index 10,
        nose index 1, and chin index 152.

    image_shape : tuple[int, ...]
        Output image shape. The first two values are interpreted as
        ``(height, width)``.

    support_mask : np.ndarray
        Boolean full-frame Sapiens2 face-and-neck mask, including mouth parts.
        Its outer boundary determines where the two jaw-end segments stop.

    margin_ratio : float, default=0.03
        Additional distance retained below the jaw, expressed as a fraction of
        the forehead-to-chin landmark distance.

    Returns
    -------
    np.ndarray
        Boolean full-frame mask with shape ``(height, width)``. True pixels are
        the semantic-mask component on the nose side of the jaw barrier, plus
        the barrier itself.

    Raises
    ------
    ValueError
        If ``support_mask`` does not match ``image_shape`` or ``margin_ratio``
        is negative.
    RuntimeError
        If landmarks have an invalid shape, do not contain the required jaw
        arc, or cannot define a forehead-to-chin orientation.

    Notes
    -----
    The forehead-to-chin vector defines the direction used by ``margin_ratio``.
    The jaw itself follows the projected landmark arc, including a protruding
    chin. If the barrier does not split the semantic mask, the function returns
    an all-true mask as a conservative fallback.
    """
    import cv2

    h, w = image_shape[:2]
    support = np.asarray(support_mask).astype(bool)
    if support.shape != (h, w):
        raise ValueError(
            'support_mask must match the first two image_shape dimensions.'
        )
    if not np.any(support):
        raise RuntimeError('Cannot split an empty semantic support mask.')
    if face_xy.ndim != 2 or face_xy.shape[1] != 2:
        raise RuntimeError('Invalid face landmarks.')
    if face_xy.shape[0] <= max(MEDIAPIPE_JAWLINE):
        raise RuntimeError('Face landmarks do not contain the jawline.')
    if margin_ratio < 0:
        raise ValueError(
            f'Invalid margin_ratio={margin_ratio!r}; expected >= 0.'
        )

    chin = face_xy[152].astype(np.float64)
    forehead = face_xy[10].astype(np.float64)
    upper = forehead - chin
    face_height = float(np.linalg.norm(upper))
    if face_height < 1.0:
        raise RuntimeError('Cannot derive face orientation from landmarks.')
    upper /= face_height

    jaw = face_xy[np.asarray(MEDIAPIPE_JAWLINE)].astype(np.float64)
    jaw -= upper * (float(margin_ratio) * face_height)

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(support.astype(np.uint8), kernel, iterations=1) > 0
    boundary_yx = np.argwhere(support & ~eroded)
    if boundary_yx.size == 0:
        return np.ones((h, w), dtype=bool)
    boundary_xy = boundary_yx[:, ::-1].astype(np.float64)

    def _nearest_boundary(point_xy: np.ndarray) -> np.ndarray:
        distances_sq = np.sum((boundary_xy - point_xy) ** 2, axis=1)
        return boundary_xy[int(np.argmin(distances_sq))]

    left_boundary = _nearest_boundary(jaw[0])
    right_boundary = _nearest_boundary(jaw[-1])
    barrier_points = np.vstack([left_boundary, jaw, right_boundary])
    barrier_points = np.rint(barrier_points).astype(np.int32)

    barrier = np.zeros((h, w), dtype=np.uint8)
    thickness = max(3, int(round(0.01 * face_height)))
    cv2.polylines(
        barrier,
        [barrier_points],
        isClosed=False,
        color=255,
        thickness=thickness,
        lineType=cv2.LINE_8,
    )

    component_count, labels = cv2.connectedComponents(
        (support & (barrier == 0)).astype(np.uint8),
        connectivity=8,
    )
    if component_count < 3:
        return np.ones((h, w), dtype=bool)

    nose_x, nose_y = face_xy[1]
    nose_x = max(0, min(w - 1, int(nose_x)))
    nose_y = max(0, min(h - 1, int(nose_y)))
    selected = int(labels[nose_y, nose_x])
    if selected == 0:
        foreground_yx = np.argwhere(labels > 0)
        if foreground_yx.size == 0:
            return np.ones((h, w), dtype=bool)
        nose_yx = np.asarray([nose_y, nose_x], dtype=np.float64)
        distances_sq = np.sum((foreground_yx - nose_yx) ** 2, axis=1)
        nearest_y, nearest_x = foreground_yx[int(np.argmin(distances_sq))]
        selected = int(labels[nearest_y, nearest_x])

    return (labels == selected) | ((barrier > 0) & support)


def eye_bbox_xyxy_from_landmarks(
    face_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float = 1.2,
) -> tuple[int, int, int, int]:
    """
    Compute an expanded eye bbox from MediaPipe face landmarks.

    Parameters
    ----------
    face_xy : np.ndarray
        Pixel-space MediaPipe face landmarks with shape ``(N, 2)``.

    image_shape : tuple[int, ...]
        Shape of the image containing the landmarks. The first two values are
        interpreted as ``(height, width)``.

    which : {'left', 'right', 'both'}
        Anatomical MediaPipe eye group to include. These values do not describe
        image/viewer-relative sides.

    expansion : float, default=1.2
        Multiplicative expansion applied to bbox width and height around its
        center.

    Returns
    -------
    tuple[int, int, int, int]
        Clipped end-exclusive bbox ``(x1, y1, x2, y2)``.

    Raises
    ------
    ValueError
        If ``which`` is unsupported.

    Notes
    -----
    The bbox uses the broad MediaPipe index ranges 33-132 and 362-462. Use
    :func:`eye_mask_from_landmarks` when the tighter eyelid contour is needed.
    """
    h, w = image_shape[:2]

    left_ids = list(range(33, 133))
    right_ids = list(range(362, 463))

    if which == 'left':
        ids = left_ids
    elif which == 'right':
        ids = right_ids
    elif which == 'both':
        ids = left_ids + right_ids
    else:
        raise ValueError(f'Invalid which={which!r}')

    pts = face_xy[np.asarray(ids)]

    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0])) + 1
    y2 = int(np.max(pts[:, 1])) + 1

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw *= expansion
    bh *= expansion

    ex1 = int(cx - bw / 2)
    ex2 = int(cx + bw / 2)
    ey1 = int(cy - bh / 2)
    ey2 = int(cy + bh / 2)

    ex1 = max(0, ex1)
    ey1 = max(0, ey1)
    ex2 = min(w, ex2)
    ey2 = min(h, ey2)

    return ex1, ey1, ex2, ey2


def eye_mask_from_landmarks(
    face_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float,
) -> np.ndarray:
    """
    Build a landmark-derived mask for one or both eyes.

    Each selected eye is represented by the convex hull of its MediaPipe eyelid
    contour. Optional expansion is implemented as morphological dilation.

    Parameters
    ----------
    face_xy : np.ndarray
        Pixel-space MediaPipe face landmarks with shape ``(N, 2)``.

    image_shape : tuple[int, ...]
        Output mask shape. The first two values are interpreted as
        ``(height, width)``.

    which : {'left', 'right', 'both'}
        Anatomical MediaPipe eye group to include. These values do not describe
        image/viewer-relative sides.

    expansion : float
        Expansion factor. Values greater than one dilate the mask by
        ``round((expansion - 1) * 8)`` pixels, with a minimum radius of one.
        Values at or below one leave the convex hull unchanged.

    Returns
    -------
    np.ndarray
        Boolean eye mask with shape ``(height, width)``.

    Notes
    -----
    Landmark coordinates are clipped to the output bounds before their convex
    hull is filled. Unsupported ``which`` values currently produce an empty
    mask rather than raising an exception.
    """
    import cv2

    h, w = image_shape[:2]

    left_ids = [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246,
    ]
    right_ids = [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398,
    ]

    def _points(ids):
        pts = []
        for i in ids:
            x, y = face_xy[int(i)]
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            pts.append([x, y])
        return np.asarray(pts, dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)

    if which in ('left', 'both'):
        pts = _points(left_ids)
        if pts.size > 0:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)

    if which in ('right', 'both'):
        pts = _points(right_ids)
        if pts.size > 0:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)

    if expansion > 1.0:
        radius = max(1, int(round((expansion - 1.0) * 8)))
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        mask = cv2.dilate(mask, k, iterations=1)

    return (mask > 0)


def eyebrow_bbox_xyxy_from_landmarks(
    face_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float = 1.2,
) -> tuple[int, int, int, int]:
    """
    Compute an expanded eyebrow bbox from MediaPipe face landmarks.

    Parameters
    ----------
    face_xy : np.ndarray
        Pixel-space MediaPipe face landmarks with shape ``(N, 2)``.

    image_shape : tuple[int, ...]
        Shape of the image containing the landmarks. The first two values are
        interpreted as ``(height, width)``.

    which : {'left', 'right', 'both'}
        Anatomical MediaPipe eyebrow group to include. These values do not
        describe image/viewer-relative sides.

    expansion : float, default=1.2
        Multiplicative expansion applied to bbox width and height around its
        center.

    Returns
    -------
    tuple[int, int, int, int]
        Clipped end-exclusive bbox ``(x1, y1, x2, y2)``.

    Raises
    ------
    ValueError
        If ``which`` is unsupported.
    """
    h, w = image_shape[:2]

    left_ids = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
    right_ids = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]

    if which == 'left':
        ids = left_ids
    elif which == 'right':
        ids = right_ids
    elif which == 'both':
        ids = left_ids + right_ids
    else:
        raise ValueError(f'Invalid which={which!r}')

    pts = face_xy[np.asarray(ids)]

    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0])) + 1
    y2 = int(np.max(pts[:, 1])) + 1

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw *= expansion
    bh *= expansion

    ex1 = int(cx - bw / 2)
    ex2 = int(cx + bw / 2)
    ey1 = int(cy - bh / 2)
    ey2 = int(cy + bh / 2)

    ex1 = max(0, ex1)
    ey1 = max(0, ey1)
    ex2 = min(w, ex2)
    ey2 = min(h, ey2)

    return ex1, ey1, ex2, ey2

def eyebrow_mask_from_landmarks(
    face_xy: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    which: str,
    expansion: float,
) -> np.ndarray:
    """
    Build a landmark-derived mask for one or both eyebrows.

    Each selected eyebrow is represented by the convex hull of its MediaPipe
    eyebrow landmarks. Optional expansion is implemented as morphological
    dilation.

    Parameters
    ----------
    face_xy : np.ndarray
        Pixel-space MediaPipe face landmarks with shape ``(N, 2)``.

    image_shape : tuple[int, ...]
        Output mask shape. The first two values are interpreted as
        ``(height, width)``.

    which : {'left', 'right', 'both'}
        Anatomical MediaPipe eyebrow group to include. These values do not
        describe image/viewer-relative sides.

    expansion : float
        Expansion factor. Values greater than one dilate the mask by
        ``round((expansion - 1) * 8)`` pixels, with a minimum radius of one.
        Values at or below one leave the convex hull unchanged.

    Returns
    -------
    np.ndarray
        Boolean eyebrow mask with shape ``(height, width)``.

    Notes
    -----
    Convex hulls only approximate visible eyebrow hair and can include nearby
    skin on arched brows, three-quarter views, or drifting landmarks.
    Unsupported ``which`` values currently produce an empty mask.
    """
    import cv2

    h, w = image_shape[:2]

    left_ids = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
    right_ids = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]

    def _points(ids):
        pts = []
        for i in ids:
            x, y = face_xy[int(i)]
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            pts.append([x, y])
        return np.asarray(pts, dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)

    if which in ('left', 'both'):
        pts = _points(left_ids)
        if pts.size > 0:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)

    if which in ('right', 'both'):
        pts = _points(right_ids)
        if pts.size > 0:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)

    if expansion > 1.0:
        radius = max(1, int(round((expansion - 1.0) * 8)))
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        mask = cv2.dilate(mask, k, iterations=1)

    return (mask > 0)


def face_landmarks_from_pose_guided_head_area(
    img_rgb: np.ndarray,
    *,
    pose_xy: np.ndarray,
    face_landmarker,
    expansion: float = 1.6,
) -> dict[str, object]:
    """
    Detect face landmarks inside a pose-guided head crop.

    A coarse head area is first derived from full-frame MediaPipe pose
    landmarks. Face detection then runs inside that crop, where the face occupies
    a larger fraction of the input. Local landmarks and their bbox are finally
    translated back to full-image coordinates.

    Parameters
    ----------
    img_rgb : np.ndarray
        Full RGB image with shape ``(height, width, 3)``.

    pose_xy : np.ndarray
        Full-image MediaPipe pose landmarks with shape ``(33, 2)`` in
        ``(x, y)`` pixel order.

    face_landmarker : object
        Initialized MediaPipe FaceLandmarker exposing ``detect``.

    expansion : float, default=1.6
        Expansion factor passed to the pose-derived head-area crop.

    Returns
    -------
    dict[str, object]
        Detection geometry containing:

        - ``head_area_rgb``: contiguous RGB head crop;
        - ``head_area_xyxy``: end-exclusive full-image crop bbox;
        - ``head_offset_xy``: crop origin ``(x, y)``;
        - ``face_xy_local``: landmarks in head-crop coordinates;
        - ``face_xy_global``: landmarks in full-image coordinates;
        - ``face_bbox_local``: end-exclusive bbox in head-crop coordinates;
        - ``face_bbox_global``: end-exclusive bbox in full-image coordinates.

    Raises
    ------
    ValueError
        If ``img_rgb`` is not an RGB image or ``pose_xy`` does not have shape
        ``(33, 2)``.
    RuntimeError
        If no valid pose-derived head crop or face landmark set can be produced.

    Notes
    -----
    If MediaPipe reports multiple faces inside the head crop,
    :func:`mp_face_landmarks` selects the one with the largest landmark bbox.
    """
    from morphalo.nodes.vision.human import crop_head_area_from_pose

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError('img_rgb must have shape (H, W, 3).')

    if pose_xy.ndim != 2 or pose_xy.shape != (33, 2):
        raise ValueError(
            f'Invalid pose_xy shape {pose_xy.shape!r}; expected (33, 2).'
        )

    head_area_rgb, x1, y1 = crop_head_area_from_pose(
        img_rgb,
        pose_xy=pose_xy,
        expansion=expansion,
    )

    face_xy_local = mp_face_landmarks(
        head_area_rgb,
        face_landmarker=face_landmarker,
    )

    face_bbox_local = face_bbox_xyxy_from_landmarks(
        face_xy_local,
        head_area_rgb.shape,
    )

    face_xy_global = offset_landmarks_xy(
        face_xy_local,
        dx=x1,
        dy=y1,
    )

    face_bbox_global = offset_bbox_xyxy(
        face_bbox_local,
        dx=x1,
        dy=y1,
    )

    h_head, w_head = head_area_rgb.shape[:2]
    head_area_xyxy = (x1, y1, x1 + w_head, y1 + h_head)

    return {
        'head_area_rgb': head_area_rgb,
        'head_area_xyxy': head_area_xyxy,
        'head_offset_xy': (x1, y1),
        'face_xy_local': face_xy_local,
        'face_xy_global': face_xy_global,
        'face_bbox_local': face_bbox_local,
        'face_bbox_global': face_bbox_global,
    }
