import numpy as np
from PIL import Image

from app.mapper import Element
from app.motion import build_rig, fixture_colours
from app.planner import Plan


def rig_and_plan(motion="climb"):
    img = Image.new("RGB", (60, 100), "black")
    m = Image.new("L", (60, 100), 0)
    m.paste(255, (20, 10, 40, 90))  # element: x 20-40, y 10-90
    els = [Element(1, "tower", "tower", (20, 10, 40, 90), m)]
    plan = Plan.model_validate({"title": "t", "duration_s": 10, "steps": [
        {"start_s": 0, "end_s": 10, "element_ids": [1], "motion": motion, "colors": ["#FF0000"], "description": "d"}]})
    return build_rig(img, els, scale=1, count=100), plan


def test_fixtures_stay_inside_the_element():
    rig, _ = rig_and_plan()
    f = rig.fx[1]
    assert len(f.ys) > 20
    assert f.xs.min() >= 20 and f.xs.max() <= 40 and f.ys.min() >= 10 and f.ys.max() <= 90


def test_climb_lights_more_fixtures_from_the_bottom_up():
    rig, plan = rig_and_plan()
    f = rig.fx[1]
    on = lambda t: fixture_colours(rig, plan, t)[:, 0] > 128
    a, b, c = on(3), on(5), on(7)
    assert a.sum() < b.sum() < c.sum()
    assert f.ys[a].min() > 50  # early on, only the bottom of the element is lit
