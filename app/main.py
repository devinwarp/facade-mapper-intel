from fastapi import FastAPI

app = FastAPI(title="facade-mapper-intel")


@app.get("/health")
def health():
    return {"ok": True}


import base64, io

from fastapi import File, Form, UploadFile
from PIL import Image

from app.mapper import map_facade
from app.planner import make_plan


@app.post("/map")
async def map_endpoint(photo: UploadFile = File(...)):
    """F1: one photo in -> labelled element masks out (PNG per element, white = element)."""
    img = Image.open(io.BytesIO(await photo.read()))
    return _elements_json(img, await map_facade(img))


def _elements_json(img, els):
    def png(m):
        b = io.BytesIO(); m.save(b, "PNG"); return base64.b64encode(b.getvalue()).decode()

    return {"width": img.width, "height": img.height, "elements": [
        {"id": e.id, "label": e.label, "kind": e.kind, "role": "dynamic", "box": e.box, "mask_png_b64": png(e.mask)}
        for e in els]}


@app.post("/plan")
async def plan_endpoint(photo: UploadFile = File(...), brief: str = Form(...), duration_s: float = Form(30)):
    """F2: photo + brief -> element map + a timed plan grounded in those elements."""
    img = Image.open(io.BytesIO(await photo.read()))
    els = await map_facade(img)
    plan = await make_plan(img, els, brief, duration_s)
    return {**_elements_json(img, els), "plan": plan.model_dump()}
