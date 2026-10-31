"""F2: architecture-aware show plan. Muse Spark reads photo + element map + brief and writes a timed plan
whose every step points at real element ids. Invalid plans are rejected and re-asked once."""
import asyncio, json
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.mapper import Element, _client, _data_url

MOTIONS = ["climb", "descend", "sweep_left_to_right", "sweep_right_to_left", "pulse", "fill", "text", "logo", "shimmer", "hold"]


class Step(BaseModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    element_ids: list[int] = Field(min_length=1)
    motion: Literal[tuple(MOTIONS)]  # type: ignore[valid-type]
    colors: list[str] = Field(description="hex colours like #00732F", min_length=1)
    text: str | None = Field(default=None, description="exact words to show, for motion=text")
    description: str = Field(description="one sentence naming the elements, e.g. 'the wave climbs the right tower'")


class Plan(BaseModel):
    title: str
    duration_s: float = Field(gt=0)
    steps: list[Step] = Field(min_length=1)

    @model_validator(mode="after")
    def _times(self):
        for s in self.steps:
            if s.end_s <= s.start_s or s.end_s > self.duration_s + 1e-6:
                raise ValueError(f"step {s.start_s}-{s.end_s}s is outside 0-{self.duration_s}s")
        return self


def check_grounding(plan: Plan, valid_ids: set[int]) -> list[str]:
    """Every step must name real element ids. Returns problems (empty = grounded)."""
    return [f"step {i} ({s.description!r}) uses unknown element ids {sorted(set(s.element_ids) - valid_ids)}"
            for i, s in enumerate(plan.steps) if not set(s.element_ids) <= valid_ids]


SYSTEM = (
    "You design light shows for a media facade. The photo is the building; the element list is its real "
    "architecture (id, label, kind, pixel box). Write a timed plan where each step moves content ON specific "
    "elements, using only element ids from the list. Motion follows the architecture: climb the towers, put a "
    "logo or text on a crown or large flat element, never across gaps. Keep text short (<= 12 characters per "
    "line) and large. Cover the whole duration with no more than 10 steps. Use only these motions: " + ", ".join(MOTIONS) + "."
)


async def make_plan(img: Image.Image, els: list[Element], brief: str, duration_s: float = 30) -> Plan:
    elist = [{"id": e.id, "label": e.label, "kind": e.kind, "box": e.box} for e in els]
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": f"Brief: {brief}\nDuration: {duration_s}s\nImage size: {img.width}x{img.height}\nElements: {json.dumps(elist)}"},
            {"type": "image_url", "image_url": {"url": _data_url(img)}},
        ]},
    ]
    valid = {e.id for e in els}
    for attempt in range(2):
        r = await _client().chat.completions.create(
            model="muse-spark-1.3", messages=msgs,
            response_format={"type": "json_schema", "json_schema": {"name": "plan", "schema": Plan.model_json_schema()}},
        )
        raw = r.choices[0].message.content
        try:
            plan = Plan.model_validate_json(raw)
            problems = check_grounding(plan, valid)
        except ValidationError as ex:
            plan, problems = None, [str(ex)[:600]]
        if not problems:
            return plan
        msgs += [{"role": "assistant", "content": raw},
                 {"role": "user", "content": "Fix these problems and return the full plan again:\n" + "\n".join(problems)}]
    raise ValueError("plan failed validation twice: " + "; ".join(problems))


if __name__ == "__main__":  # python -m app.planner photo.png "brief" [elements.json]
    import sys
    from app.mapper import map_facade
    im = Image.open(sys.argv[1])
    els = asyncio.run(map_facade(im))
    plan = asyncio.run(make_plan(im, els, sys.argv[2]))
    print(plan.model_dump_json(indent=1))
