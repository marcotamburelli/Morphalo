import sys
from types import SimpleNamespace

import numpy as np

from morphalo.nodes.vision.face_landmarks import (mp_face_landmarks,
                                                  mp_face_landmarks_xyz,
                                                  mp_face_landmarks_xyz_with_transform)


class _FakeLandmarker:

    def __init__(self, faces, transforms=None):
        self._faces = faces
        self._transforms = transforms or []

    def detect(self, image):
        return SimpleNamespace(
            face_landmarks=self._faces,
            facial_transformation_matrixes=self._transforms,
        )


def _landmark(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def test_xyz_scales_depth_and_selects_largest_face(monkeypatch):
    fake_mediapipe = SimpleNamespace(
        Image=lambda **kwargs: kwargs,
        ImageFormat=SimpleNamespace(SRGB='srgb'),
    )
    monkeypatch.setitem(sys.modules, 'mediapipe', fake_mediapipe)

    small_face = [
        _landmark(0.4, 0.4, -0.1),
        _landmark(0.6, 0.6, 0.1),
    ]
    large_face = [
        _landmark(-0.1, 0.2, -0.25),
        _landmark(0.9, 0.8, 0.5),
    ]
    landmarker = _FakeLandmarker([small_face, large_face])
    image = np.zeros((50, 100, 3), dtype=np.uint8)

    xyz = mp_face_landmarks_xyz(image, face_landmarker=landmarker)
    xy = mp_face_landmarks(image, face_landmarker=landmarker)

    np.testing.assert_allclose(
        xyz,
        [[0.0, 10.0, -25.0], [90.0, 40.0, 50.0]],
    )
    np.testing.assert_array_equal(xy, [[0, 10], [90, 40]])


def test_xyz_with_transform_returns_matrix_for_selected_face(monkeypatch):
    fake_mediapipe = SimpleNamespace(
        Image=lambda **kwargs: kwargs,
        ImageFormat=SimpleNamespace(SRGB='srgb'),
    )
    monkeypatch.setitem(sys.modules, 'mediapipe', fake_mediapipe)

    small_face = [_landmark(0.4, 0.4, 0), _landmark(0.6, 0.6, 0)]
    large_face = [_landmark(0.1, 0.1, 0), _landmark(0.9, 0.9, 0)]
    transforms = [np.eye(4), np.eye(4) * 2]
    landmarker = _FakeLandmarker([small_face, large_face], transforms)

    _, transform = mp_face_landmarks_xyz_with_transform(
        np.zeros((100, 100, 3), dtype=np.uint8),
        face_landmarker=landmarker,
    )

    np.testing.assert_array_equal(transform, transforms[1])
