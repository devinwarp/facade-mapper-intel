"""Call each Meta model once to see what works from this machine/region.

Run: uv run python scripts/smoke.py   (needs MODEL_API_KEY)
The SAM request shape is a GUESS from secondary docs; check dev.meta.ai/docs/sam/segmenting.
"""
import base64, os, sys, httpx
from openai import OpenAI

KEY = os.environ.get("MODEL_API_KEY") or sys.exit("set MODEL_API_KEY")
BASE = "https://api.meta.ai/v1"
client = OpenAI(base_url=BASE, api_key=KEY)


def step(name, fn):
    try:
        print(f"[ok]   {name}: {fn()}")
    except Exception as e:  # smoke test: report everything, stop on nothing
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}")


def spark():
    r = client.chat.completions.create(
        model="muse-spark-1.3", messages=[{"role": "user", "content": "Say ok."}]
    )
    return r.choices[0].message.content


def image():
    r = client.images.generate(model="muse-image-1.0", prompt="a glass tower at dusk", size="1024x1024")
    return f"{len(r.data[0].b64_json or '')} b64 chars"


def sam():
    img = sys.argv[1] if len(sys.argv) > 1 else None
    if not img:
        return "skipped (pass a facade photo path as argv[1])"
    b64 = base64.b64encode(open(img, "rb").read()).decode()
    r = httpx.post(
        f"{BASE}/responses",
        headers={"Authorization": f"Bearer {KEY}"},
        timeout=120,
        json={"model": "sam-3.1", "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
            {"type": "input_text", "text": "vertical fins"},
        ]}]},
    )
    r.raise_for_status()
    return r.text[:300]


step("muse-spark", spark)
step("muse-image", image)
step("sam-3.1", sam)
