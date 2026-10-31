import numpy as np
from PIL import Image

from app.mapper import Element
from app.motion import frame_at, prepare
from app.planner import Plan


def setup(motion="climb"):
    m = Image.new("L", (60, 100), 0)
    m.paste(255, (20, 10, 40, 90))  # element: x 20-40, y 10-90
    els = [Element(1, "tower", "tower", (20, 10, 40, 90), m)]
    plan = Plan.model_validate({"title": "t", "duration_s": 10, "steps": [
        {"start_s": 0, "end_s": 10, "element_ids": [1], "motion": motion, "colors": ["#FF0000"], "description": "d"}]})
    base = np.zeros((100, 60, 3), np.float32)
    base[::4] = 255  # horizontal stripes = edges for the fixture finder
    return base, plan, prepare(els, base)


def test_climb_rises_and_stays_inside_the_element():
    base, plan, prep = setup()
    inside = prep[1].mask
    bright = lambda t: (frame_at(base, plan, prep, t)[..., 0] > 200) & (frame_at(base, plan, prep, t)[..., 1] < 60)
    red = [bright(t) for t in (2, 5, 8)]
    counts = [r.sum() for r in red]
    assert counts[0] < counts[1] < counts[2]          # more lit as the front climbs
    assert not (red[2] & ~inside).any()               # nothing lit outside the element
    ys = np.where(red[0].any(axis=1))[0]
    assert ys.min() > 50                              # early on, only the bottom is lit
