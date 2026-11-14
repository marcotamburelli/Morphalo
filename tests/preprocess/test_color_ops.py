"""Statistical color matching with independent sampling and application regions."""
import numpy as np
import pytest
from PIL import Image

from morphalo.nodes.preprocess.utils.color_ops import match_lab_color_statistics


def test_statistics_mask_excludes_artifacts_without_restricting_application():
    y, x = np.mgrid[:20, :24]
    source = np.stack((120 + x, 60 + y, 40 + x), axis=-1).astype(np.uint8)
    reference = np.stack((50 + x, 100 + y, 90 + x), axis=-1).astype(np.uint8)
    samples = np.zeros((20, 24), bool)
    samples[5:15, 6:18] = True
    alpha = np.full(samples.shape, 255, np.uint8)
    alpha[0] = 0
    rgba = Image.fromarray(np.dstack((source, alpha)))
    clean = match_lab_color_statistics(rgba, Image.fromarray(reference), strength=0.7, statistics_mask=samples)
    contaminated = reference.copy()
    contaminated[~samples] = [255, 0, 255]
    actual = match_lab_color_statistics(rgba, Image.fromarray(contaminated), strength=0.7, statistics_mask=samples)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(clean))
    np.testing.assert_array_equal(np.asarray(actual)[:, :, 3], alpha)
    # A visible pixel outside the statistics region still receives correction.
    assert not np.array_equal(np.asarray(actual)[2, 2, :3], source[2, 2])


def test_color_matching_empty_statistics_mask_is_identity():
    source = Image.new('RGBA', (16, 12), (190, 80, 50, 170))
    reference = Image.new('RGB', source.size, (20, 90, 120))
    actual = match_lab_color_statistics(source, reference, strength=1., statistics_mask=np.zeros((12, 16), bool))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(source))
    with pytest.raises(ValueError, match='statistics_mask'):
        match_lab_color_statistics(source, reference, strength=1., statistics_mask=np.ones((16, 12), bool))
