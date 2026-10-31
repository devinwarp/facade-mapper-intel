import pytest
from pydantic import ValidationError

from app.planner import Plan, check_grounding


def plan(ids, end=5, dur=10):
    return Plan.model_validate({"title": "t", "duration_s": dur, "steps": [
        {"start_s": 0, "end_s": end, "element_ids": ids, "motion": "climb", "colors": ["#00732F"], "description": "d"}]})


def test_grounded_plan_passes():
    assert check_grounding(plan([1, 2]), {1, 2, 3}) == []


def test_unknown_element_is_flagged():
    assert "99" in check_grounding(plan([1, 99]), {1, 2})[0]


def test_step_past_duration_rejected():
    with pytest.raises(ValidationError):
        plan([1], end=11)
