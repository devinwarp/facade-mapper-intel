import numpy as np

from app.survey import Survey, apply, elevations, homography, pick_elevation, quad_from_mask


def test_homography_maps_corners():
    src = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], float)
    dst = np.array([[5, 5], [15, 7], [14, 30], [4, 28]], float)
    x, y = apply(homography(src, dst), src[:, 0], src[:, 1])
    assert np.allclose(np.stack([x, y], 1), dst, atol=1e-6)


def test_elevations_split_at_gaps_and_skip_short_bands():
    cx = np.r_[np.full(10, 100.0), np.full(10, 600.0), np.full(10, 1100.0)]
    cy = np.r_[np.linspace(0, 900, 10), np.linspace(0, 900, 10), np.linspace(900, 950, 10)]  # 3rd = podium band
    sv = Survey(cx, cy, 10, 20, (1200, 1000))
    r = elevations(sv)
    assert len(r) == 2
    assert pick_elevation(r, 0.1) in (0, 1)


def test_quad_from_mask():
    m = np.zeros((100, 60), bool); m[10:90, 20:40] = True
    q = quad_from_mask(m)
    assert q[0].tolist() == [20, 10] and q[2].tolist() == [39, 89]
