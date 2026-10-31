"""F4: element-driven motion. Our own code (no model): every plan step is drawn only inside its elements'
masks and moves along each element's own extent, so motion follows the architecture."""
import asyncio, subprocess, sys
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.mapper import Element
from app.planner import Plan, Step

DIM = 0.35   # how dark the night photo is under the light
FADE_S = 0.4  # fade in/out at the ends of a step
BAND = 0.18   # colour band height as a fraction of the element extent


def _hex(c: str) -> np.ndarray:
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


@dataclass
class Prepared:
    mask: np.ndarray  # bool HxW
    y0: int; y1: int; x0: int; x1: int  # extent of the mask itself, not the Spark box
    light: np.ndarray  # float HxW 0..1: where fixtures + their glow can emit


def fixtures(base: np.ndarray, mask: np.ndarray, vertical: bool, keep: float = 0.15) -> np.ndarray:
    """Virtual linear fixtures: the element's own strongest edges across its dominant direction.
    Light exists only on these lines (plus glow); the surface between them stays dark.
    ponytail: stand-in for the real survey (fixture layout); swap in surveyed lines when available."""
    g = base.mean(axis=2)
    d = np.zeros_like(g)
    if vertical:
        d[:, 1:] = np.abs(g[:, 1:] - g[:, :-1])
    else:
        d[1:] = np.abs(g[1:] - g[:-1])
    thr = np.quantile(d[mask], 1 - keep) if mask.any() else 1
    line = Image.fromarray(((d >= thr) & mask).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))
    glow = np.asarray(line.filter(ImageFilter.GaussianBlur(2.5)), np.float32) / 255
    return np.clip(np.asarray(line, np.float32) / 255 + 0.7 * glow, 0, 1) * mask


def prepare(els: list[Element], base: np.ndarray) -> dict[int, Prepared]:
    out = {}
    for e in els:
        m = np.asarray(e.mask) > 127
        ys, xs = np.where(m)
        if len(ys):
            out[e.id] = Prepared(m, ys.min(), ys.max() + 1, xs.min(), xs.max() + 1, fixtures(base, m, e.kind == "fins"))
    return out


def _stripes(axis: np.ndarray, colors: list[str], start: float, extent: float) -> np.ndarray:
    """Colour by distance behind `start` along the axis, one band per colour, cycling."""
    cols = np.stack([_hex(c) for c in colors])
    k = np.floor(np.abs(axis - start) / max(extent * BAND, 1)).astype(int) % len(cols)
    return cols[k]


def lit(step: Step, p: Prepared, q: float, t: float, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (alpha HxW in 0..1, colour HxWx3) for one element at step progress q (0..1), time t."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    h, w = p.y1 - p.y0, p.x1 - p.x0
    soft = lambda d, s: np.clip(d / s + 0.5, 0, 1)  # soft edge, s = edge width in px
    mo = step.motion
    if mo == "climb":     # front travels bottom -> top; lit behind it
        front = p.y1 - q * (h + 1)
        return soft(yy - front, 6), _stripes(yy, step.colors, front, h)
    if mo == "descend":
        front = p.y0 + q * (h + 1)
        return soft(front - yy, 6), _stripes(yy, step.colors, front, h)
    if mo == "sweep_left_to_right":
        front = p.x0 + q * (w + 1)
        return soft(front - xx, 6), _stripes(xx, step.colors, front, w)
    if mo == "sweep_right_to_left":
        front = p.x1 - q * (w + 1)
        return soft(xx - front, 6), _stripes(xx, step.colors, front, w)
    if mo == "pulse":
        a = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * q)
        return np.full((H, W), a, np.float32), np.broadcast_to(_hex(step.colors[0]), (H, W, 3))
    if mo == "shimmer":
        a = 0.55 + 0.45 * np.sin((yy + xx) / 14.0 - t * 6)
        return a.astype(np.float32), np.broadcast_to(_hex(step.colors[0]), (H, W, 3))
    if mo == "text" and step.text:
        return _text_alpha(step.text, p, shape), np.broadcast_to(_hex(step.colors[0]), (H, W, 3))
    # fill / hold / logo (no logo asset yet): whole element, colours as static bands
    return np.ones((H, W), np.float32), _stripes(yy, step.colors, p.y1, h)


def _text_alpha(text: str, p: Prepared, shape: tuple[int, int]) -> np.ndarray:
    """Largest bold text that fits the element extent, centred."""
    w, h = p.x1 - p.x0, p.y1 - p.y0
    size = max(8, min(h, w))
    while size > 8:
        f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size, index=1)
        l, t, r, b = f.getbbox(text)
        if r - l <= w * 0.92 and b - t <= h * 0.8:
            break
        size -= 2
    img = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(img).text((p.x0 + w / 2, p.y0 + h / 2), text, font=f, fill=255, anchor="mm")
    return np.asarray(img, np.float32) / 255


def frame_at(base: np.ndarray, plan: Plan, prep: dict[int, Prepared], t: float) -> np.ndarray:
    """Composite every active step at time t over the dimmed photo."""
    out = base * DIM
    for s in plan.steps:
        if not (s.start_s <= t < s.end_s):
            continue
        q = (t - s.start_s) / (s.end_s - s.start_s)
        env = min(1.0, (t - s.start_s) / FADE_S, (s.end_s - t) / FADE_S)  # fade envelope
        for eid in s.element_ids:
            p = prep.get(eid)
            if p is None:
                continue
            a, col = lit(s, p, q, t, base.shape[:2])
            a = (a * p.light * env)[..., None]
            out = out * (1 - a) + col * a  # light only where a fixture is
    return out


def render(img: Image.Image, els: list[Element], plan: Plan, out_path: str, fps: int = 24) -> int:
    """Write an H.264 preview of the plan composited on the photo. Returns frame count."""
    w, h = img.width // 2 * 2, img.height // 2 * 2  # h264 needs even sizes
    base = np.asarray(img.convert("RGB").crop((0, 0, w, h)), np.float32)
    prep = prepare(els, base)
    n = int(plan.duration_s * fps)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path],
        stdin=subprocess.PIPE)
    for i in range(n):
        ff.stdin.write(frame_at(base, plan, prep, i / fps).clip(0, 255).astype(np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    return n


if __name__ == "__main__":  # python -m app.motion photo.png "brief" out.mp4 [duration_s]
    from app.mapper import map_facade
    from app.planner import make_plan
    im = Image.open(sys.argv[1])
    dur = float(sys.argv[4]) if len(sys.argv) > 4 else 20

    import os, pickle
    cache = sys.argv[3] + ".pkl"  # reuse map+plan so motion can be iterated without new model calls

    async def go():
        els = await map_facade(im)
        return els, await make_plan(im, els, sys.argv[2], dur)

    els, plan = pickle.load(open(cache, "rb")) if os.path.exists(cache) else asyncio.run(go())
    pickle.dump((els, plan), open(cache, "wb"))
    for e in els: print(e.id, e.kind, e.label)
    for st in plan.steps: print(st.start_s, st.end_s, st.element_ids, st.motion, st.colors[:2], st.text)
    print("frames:", render(im, els, plan, sys.argv[3]))
