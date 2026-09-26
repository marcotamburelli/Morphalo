import numpy as np


def _detect_largest_face(
    img_rgb: np.ndarray,
    *,
    face_landmarker,
) -> tuple[np.ndarray, int, object]:
    """
    Detect MediaPipe 3D face landmarks and select the largest detected face.

    The normalized MediaPipe coordinates are converted to pixel-equivalent
    coordinates: x is scaled by image width, y by image height, and relative z
    by image width, following MediaPipe's coordinate convention. The projected
    x and y coordinates are clipped to the image bounds. When multiple faces
    are present, the one with the largest projected landmark bbox is returned.

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB image with shape ``(height, width, 3)``.

    face_landmarker : object
        Initialized MediaPipe FaceLandmarker exposing ``detect``.

    Returns
    -------
    tuple[np.ndarray, int, object]
        Floating-point ``(N, 3)`` coordinates, selected face index, and raw
        MediaPipe result. The z coordinate is relative depth in
        pixel-equivalent units; smaller values are closer to the camera.

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

    best_xyz = None
    best_area = None

    best_index = None

    for index, landmarks in enumerate(face_landmarks_list):
        face_xyz = np.empty((len(landmarks), 3), dtype=np.float32)

        for i, lm in enumerate(landmarks):
            x = np.clip(float(lm.x) * w, 0.0, float(w - 1))
            y = np.clip(float(lm.y) * h, 0.0, float(h - 1))
            z = float(lm.z) * w
            face_xyz[i] = (x, y, z)

        face_xy = np.rint(face_xyz[:, :2]).astype(np.int32)
        x1 = np.min(face_xy[:, 0])
        y1 = np.min(face_xy[:, 1])
        x2 = np.max(face_xy[:, 0]) + 1
        y2 = np.max(face_xy[:, 1]) + 1

        if x2 <= x1 or y2 <= y1:
            continue

        area = float((x2 - x1) * (y2 - y1))

        if best_area is None or area > best_area:
            best_area = area
            best_xyz = face_xyz
            best_index = index

    if best_xyz is None or best_index is None:
        raise RuntimeError('Could not derive valid face landmarks.')

    return best_xyz, best_index, result


def mp_face_landmarks_xyz(
    img_rgb: np.ndarray,
    *,
    face_landmarker,
) -> np.ndarray:
    """
    Detect the largest face and return its XYZ landmarks.

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB image with shape ``(height, width, 3)``.
    face_landmarker : object
        Initialized MediaPipe FaceLandmarker exposing ``detect``.

    Returns
    -------
    np.ndarray
        Floating-point coordinates with shape ``(N, 3)`` in ``(x, y, z)``
        order. X and Y are image pixels; Z is relative depth scaled by image
        width according to MediaPipe's coordinate convention.

    Raises
    ------
    RuntimeError
        If MediaPipe detects no usable face landmarks.
    """
    face_xyz, _, _ = _detect_largest_face(
        img_rgb=img_rgb,
        face_landmarker=face_landmarker,
    )

    return face_xyz


def mp_face_landmarks_xyz_with_transform(
    img_rgb: np.ndarray,
    *,
    face_landmarker,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect the largest face and return its XYZ landmarks and pose transform.

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB image with shape ``(height, width, 3)``.
    face_landmarker : object
        FaceLandmarker configured to emit facial transformation matrices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Pixel-equivalent ``(N, 3)`` landmarks and the corresponding ``(4, 4)``
        MediaPipe facial transformation matrix.

    Raises
    ------
    RuntimeError
        If no matching facial transformation matrix is available.
    """
    face_xyz, face_index, result = _detect_largest_face(
        img_rgb=img_rgb,
        face_landmarker=face_landmarker,
    )
    matrices = result.facial_transformation_matrixes or []

    if face_index >= len(matrices):
        raise RuntimeError(
            'MediaPipe did not return a facial transformation matrix.'
        )

    transform = np.asarray(matrices[face_index], dtype=np.float32)

    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError('MediaPipe returned an invalid face transform.')

    return face_xyz, transform


def mp_face_landmarks(
    img_rgb: np.ndarray,
    *,
    face_landmarker,
) -> np.ndarray:
    """
    Detect MediaPipe face landmarks and select the largest detected face.

    This compatibility helper projects :func:`mp_face_landmarks_xyz` onto the
    image plane and rounds the result to integer pixel coordinates.

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
    face_xyz = mp_face_landmarks_xyz(
        img_rgb=img_rgb,
        face_landmarker=face_landmarker,
    )

    return np.rint(face_xyz[:, :2]).astype(np.int32)
