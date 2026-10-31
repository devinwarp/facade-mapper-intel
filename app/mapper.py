"""F1: read the building. Muse Spark proposes element boxes, SAM 3.1 outlines them."""
import asyncio, base64, io, json, os
from dataclasses import dataclass

from meta_sam_parser import (FramePrompt, PromptBox, PromptObject, build_object_prompt,
                             decode_mask_to_raster, image_segmentation_format, parse_responses_stream)
from openai import AsyncOpenAI
from PIL import Image, ImageChops

BASE = "https://api.meta.ai/v1"
MAX_OBJECTS = 16  # SAM limit per request

SCHEMA = {
    "type": "object",
    "properties": {"elements": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "kind": {"type": "string", "enum": ["crown", "tower", "podium", "setback", "band", "fins", "glass", "other"]},
            "x1": {"type": "integer"}, "y1": {"type": "integer"},
            "x2": {"type": "integer"}, "y2": {"type": "integer"},
        },
        "required": ["label", "kind", "x1", "y1", "x2", "y2"],
        "additionalProperties": False,
    }}},
    "required": ["elements"],
    "additionalProperties": False,
}

PROMPT = (
    "You are mapping a building facade for a media-facade lighting designer. List the distinct, "
    "architecturally meaningful elements visible (e.g. crown/top, each tower or setback block, podium, "
    "a representative horizontal band or vertical fin run, glazed core). At most {n}. Each element gets a "
    "short unique label, a kind, and a tight bounding box in coordinates normalised to 0-1000 "
    "(x1,y1 = top-left, x2,y2 = bottom-right). Ignore sky, trees, street and foreground clutter."
)


@dataclass
class Element:
    id: int
    label: str
    kind: str
    box: tuple[int, int, int, int]  # pixels
    mask: Image.Image | None = None  # L-mode, full image size, 255 = element


def _data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=BASE, api_key=os.environ["MODEL_API_KEY"])


async def propose(img: Image.Image, n: int = 12) -> list[Element]:
    r = await _client().chat.completions.create(
        model="muse-spark-1.3",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(n=n)},
            {"type": "image_url", "image_url": {"url": _data_url(img)}},
        ]}],
        response_format={"type": "json_schema", "json_schema": {"name": "elements", "schema": SCHEMA, "strict": True}},
    )
    w, h = img.size
    out = []
    for i, e in enumerate(json.loads(r.choices[0].message.content)["elements"][:n]):
        x1, x2 = sorted((e["x1"], e["x2"])); y1, y2 = sorted((e["y1"], e["y2"]))
        box = (max(0, x1 * w // 1000), max(0, y1 * h // 1000), min(w, x2 * w // 1000), min(h, y2 * h // 1000))
        if box[2] - box[0] > 2 and box[3] - box[1] > 2:
            out.append(Element(i + 1, e["label"], e["kind"], box))
    return out


async def outline(img: Image.Image, els: list[Element]) -> list[Element]:
    """SAM 3.1 box prompts -> one full-size mask per element."""
    w, h = img.size
    objs = [PromptObject(id=e.id, prompts=[FramePrompt(box=PromptBox(*e.box))]) for e in els]
    prompt = build_object_prompt(size=(w, h), objects=objs)
    stream = await _client().responses.create(
        model="sam-3.1", stream=True,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": _data_url(img)},
        ]}],
    )
    result = await parse_responses_stream(stream, image_segmentation_format()).final_result()
    by_id = {e.id: e for e in els}
    for rec in result.records:
        if type(rec).__name__ != "SegmentationMaskRecord":
            continue
        el = by_id.get(int(rec.object_id))
        if el is None:
            continue
        m = rec.mask
        tile = Image.frombytes("L", (m.width, m.height), bytes(v * 255 for v in decode_mask_to_raster(m)))
        full = Image.new("L", (w, h), 0)
        full.paste(tile, (int(rec.bounds.left), int(rec.bounds.top)))
        el.mask = full
    return els


async def concept_masks(img: Image.Image, text: str) -> list["np.ndarray"]:
    """SAM 3.1 concept prompt (one short noun phrase) -> one bool mask per match, full image size."""
    import numpy as np
    stream = await _client().responses.create(
        model="sam-3.1", stream=True,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": text},
            {"type": "input_image", "image_url": _data_url(img)},
        ]}],
    )
    res = await parse_responses_stream(stream, image_segmentation_format()).final_result()
    out = []
    for rec in res.records:
        if type(rec).__name__ != "SegmentationMaskRecord":
            continue
        m = rec.mask
        tile = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width) > 0
        full = np.zeros((img.height, img.width), bool)
        t, l = int(rec.bounds.top), int(rec.bounds.left)
        full[t:t + m.height, l:l + m.width] = tile
        out.append(full)
    return out


async def tower_mask(img: Image.Image) -> "np.ndarray":
    """The main tower's silhouette: the largest 'skyscraper' match (clean of annexes, unlike element kinds)."""
    ms = await concept_masks(img, "skyscraper")
    if not ms:
        raise ValueError("SAM found no skyscraper in the photo")
    return max(ms, key=lambda m: m.sum())


REGION_KINDS = {"band", "fins", "glass"}  # repeated detail: not a single object SAM can outline


def clip_regions(img: Image.Image, regions: list[Element], solids: list[Element]) -> None:
    """Region mask = its box clipped to the solid it overlaps most.
    ponytail: box-in-parent is coarse; per-band masks via point prompts or edge detection later."""
    for r in regions:
        box = Image.new("L", img.size, 0)
        box.paste(255, r.box)
        best = max(solids, key=lambda s: sum(ImageChops.multiply(s.mask, box).histogram()[128:]), default=None)
        r.mask = ImageChops.multiply(best.mask, box) if best else None


async def map_facade(img: Image.Image) -> list[Element]:
    els = await propose(img)
    solids = [e for e in els if e.kind not in REGION_KINDS]
    await outline(img, solids)
    solids = [e for e in solids if e.mask is not None]
    regions = [e for e in els if e.kind in REGION_KINDS]
    clip_regions(img, regions, solids)
    return solids + [e for e in regions if e.mask is not None]


if __name__ == "__main__":  # python -m app.mapper photo.png outdir
    import sys
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    im = Image.open(src)
    els = asyncio.run(map_facade(im))
    for e in els:
        e.mask.save(f"{outdir}/{e.id}_{e.kind}.png")
        print(e.id, e.kind, e.label, e.box)
