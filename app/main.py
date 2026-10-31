from fastapi import FastAPI

app = FastAPI(title="facade-mapper-intel")


@app.get("/health")
def health():
    return {"ok": True}


import base64, io

from fastapi import File, UploadFile
from PIL import Image

from app.mapper import map_facade


@app.post("/map")
async def map_endpoint(photo: UploadFile = File(...)):
    """F1: one photo in -> labelled element masks out (PNG per element, white = element)."""
    img = Image.open(io.BytesIO(await photo.read()))
    els = await map_facade(img)

    def png(m):
        b = io.BytesIO(); m.save(b, "PNG"); return base64.b64encode(b.getvalue()).decode()

    return {"width": img.width, "height": img.height, "elements": [
        {"id": e.id, "label": e.label, "kind": e.kind, "role": "dynamic", "box": e.box, "mask_png_b64": png(e.mask)}
        for e in els]}
