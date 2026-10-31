"""F4: element-driven motion on virtual light fixtures. Own code, no model.

From a photo alone we place linear fixtures on every element (spacing copied from a real surveyed facade),
then drive each fixture's colour from the plan. Light exists only on fixtures, as on a real building."""
import asyncio, subprocess, sys
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.mapper import Element
from app.planner import Plan, Step

# Fixture spec measured on the Wasl Tower UV survey (11682x10980 canvas): 7,140 dynamic fixtures, each a
# 10x20 px dash, 28 px pitch along the dash (runs of up to 16), ~68 px typical gap between columns.
FIXTURES = 7140  # real dynamic fixture count on that tower: default density for a tower-scale facade
ALONG = 1.4      # pitch along a dash, in dash lengths (28/20)
ACROSS = 3.4     # pitch between columns, in dash lengths (68/20)
DIM = 0.35    # how dark the night photo is under the light
FADE_S = 0.4
BAND = 0.18   # colour band height as a fraction of the element extent
GLOW = 3      # glow blur radius in output px


@dataclass
class Fx:
    """Fixtures of one element, in output pixels."""
    ys: np.ndarray; xs: np.ndarray  # centres
    y0: float; y1: float; x0: float; x1: float  # extent of the element
    vertical: bool


def _hex(c: str) -> np.ndarray:
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def place(mask: np.ndarray, vertical: bool, L: float) -> tuple[Fx, float, float]:
    """Regular grid of dashes of length L inside the mask. Returns (fixtures, dash length, dash thickness).
    ponytail: straight grid; follow curved slab edges / real mullions later."""
    pitch_along, pitch_across = ALONG * L, ACROSS * L
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if vertical:
        gx, gy = np.meshgrid(np.arange(x0 + pitch_across / 2, x1, pitch_across), np.arange(y0 + pitch_along / 2, y1, pitch_along))
    else:
        gx, gy = np.meshgrid(np.arange(x0 + pitch_along / 2, x1, pitch_along), np.arange(y0 + pitch_across / 2, y1, pitch_across))
    gx, gy = gx.ravel(), gy.ravel()
    keep = mask[np.clip(gy.astype(int), 0, mask.shape[0] - 1), np.clip(gx.astype(int), 0, mask.shape[1] - 1)]
    return Fx(gy[keep], gx[keep], y0, y1, x0, x1, vertical), L, max(1.0, L / 2)


def is_vertical(base: np.ndarray, mask: np.ndarray) -> bool:
    """Fixtures run along the dominant line direction of the element (mullions vs slab edges)."""
    g = base.mean(axis=2)
    dx = np.abs(g[:, 1:] - g[:, :-1])[mask[:, 1:]].mean() if mask.any() else 0
    dy = np.abs(g[1:] - g[:-1])[mask[1:]].mean() if mask.any() else 0
    return bool(dx > dy)


@dataclass
class Rig:
    base: np.ndarray            # dimmed photo, float HxWx3
    fx: dict[int, Fx]
    slices: dict[int, slice]    # element id -> range in the flat fixture arrays
    py: np.ndarray; px: np.ndarray; owner: np.ndarray  # splat pixels and the fixture that owns each
    n: int


def build_rig(img: Image.Image, els: list[Element], scale: float | None = None, count: int = FIXTURES) -> Rig:
    solids = [e for e in els if e.mask is not None]
    tallest = max(np.ptp(np.where(np.asarray(e.mask) > 127)[0]) for e in solids)
    scale = scale or max(1.0, 1600 / tallest)  # upsample so fine fixtures are visible
    W, H = int(img.width * scale) // 2 * 2, int(img.height * scale) // 2 * 2
    photo = img.convert("RGB").resize((W, H), Image.LANCZOS)
    small = np.asarray(img.convert("RGB"), np.float32)
    base = np.asarray(photo, np.float32)
    union = np.zeros((img.height, img.width), bool)
    for e in solids:
        union |= np.asarray(e.mask) > 127
    L = float(np.sqrt(union.sum() * scale**2 / (ALONG * ACROSS * count)))  # dash length that yields ~count fixtures
    fx, slices, owner_list, py_list, px_list, off = {}, {}, [], [], [], 0
    for e in solids:
        m_small = np.asarray(e.mask) > 127
        m = np.asarray(e.mask.resize((W, H), Image.NEAREST)) > 127
        if not m.any():
            continue
        f, L, t = place(m, is_vertical(small, m_small), L)
        if not len(f.ys):
            continue
        h, w = (L, t) if f.vertical else (t, L)
        hh, ww = max(1, int(round(h))), max(1, int(round(w)))
        dy, dx = np.meshgrid(np.arange(hh) - hh // 2, np.arange(ww) - ww // 2, indexing="ij")
        dy, dx = dy.ravel(), dx.ravel()
        cy, cx = f.ys.astype(int), f.xs.astype(int)
        py_list.append(np.clip(cy[:, None] + dy[None], 0, H - 1).ravel())
        px_list.append(np.clip(cx[:, None] + dx[None], 0, W - 1).ravel())
        owner_list.append(np.repeat(np.arange(len(cy)) + off, len(dy)))
        fx[e.id] = f
        slices[e.id] = slice(off, off + len(cy))
        off += len(cy)
    return Rig(base * DIM, fx, slices, np.concatenate(py_list), np.concatenate(px_list), np.concatenate(owner_list), off)


def _stripes(axis, colors, start, extent):
    cols = np.stack([_hex(c) for c in colors])
    k = np.floor(np.abs(axis - start) / max(extent * BAND, 1)).astype(int) % len(cols)
    return cols[k]


def _text_img(text: str, f: Fx, shape: tuple[int, int]) -> np.ndarray:
    w, h = f.x1 - f.x0, f.y1 - f.y0
    size = max(8, int(min(h, w)))
    while size > 8:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size, index=1)
        l, t, r, b = font.getbbox(text)
        if r - l <= w * 0.92 and b - t <= h * 0.8:
            break
        size -= 2
    img = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(img).text(((f.x0 + f.x1) / 2, (f.y0 + f.y1) / 2), text, font=font, fill=255, anchor="mm")
    return np.asarray(img, np.float32) / 255


def lit(step: Step, f: Fx, q: float, t: float, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(alpha N, colour Nx3) for one element's fixtures at step progress q, time t."""
    ys, xs, n = f.ys, f.xs, len(f.ys)
    h, w = f.y1 - f.y0, f.x1 - f.x0
    soft = lambda d, s: np.clip(d / s + 0.5, 0, 1)
    one = lambda a: np.full(n, a, np.float32)
    col0 = np.broadcast_to(_hex(step.colors[0]), (n, 3))
    mo = step.motion
    if mo == "climb":
        front = f.y1 - q * (h + 1); return soft(ys - front, 6), _stripes(ys, step.colors, front, h)
    if mo == "descend":
        front = f.y0 + q * (h + 1); return soft(front - ys, 6), _stripes(ys, step.colors, front, h)
    if mo == "sweep_left_to_right":
        front = f.x0 + q * (w + 1); return soft(front - xs, 6), _stripes(xs, step.colors, front, w)
    if mo == "sweep_right_to_left":
        front = f.x1 - q * (w + 1); return soft(xs - front, 6), _stripes(xs, step.colors, front, w)
    if mo == "pulse":
        return one(0.5 + 0.5 * np.sin(2 * np.pi * 3 * q)), col0
    if mo == "shimmer":
        return (0.55 + 0.45 * np.sin((ys + xs) / 14.0 - t * 6)).astype(np.float32), col0
    if mo == "text" and step.text:
        img = _text_img(step.text, f, shape)
        return img[np.clip(ys.astype(int), 0, shape[0] - 1), np.clip(xs.astype(int), 0, shape[1] - 1)], col0
    return one(1), _stripes(ys, step.colors, f.y1, h)  # fill / hold / logo


def fixture_colours(rig: Rig, plan: Plan, t: float) -> np.ndarray:
    """Colour of every fixture at time t (Nx3, 0 = off)."""
    col = np.zeros((rig.n, 3), np.float32)
    shape = rig.base.shape[:2]
    for s in plan.steps:
        if not (s.start_s <= t < s.end_s):
            continue
        q = (t - s.start_s) / (s.end_s - s.start_s)
        env = min(1.0, (t - s.start_s) / FADE_S, (s.end_s - t) / FADE_S)
        for eid in s.element_ids:
            f = rig.fx.get(eid)
            if f is None:
                continue
            a, c = lit(s, f, q, t, shape)
            sl = rig.slices[eid]
            a = (a * env)[:, None]
            col[sl] = col[sl] * (1 - a) + c * a
    return col


def frame_at(rig: Rig, plan: Plan, t: float) -> np.ndarray:
    layer = np.zeros_like(rig.base)
    layer[rig.py, rig.px] = fixture_colours(rig, plan, t)[rig.owner]
    glow = np.asarray(Image.fromarray(layer.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(GLOW)), np.float32)
    return rig.base + layer + 1.6 * glow


def render(img: Image.Image, els: list[Element], plan: Plan, out_path: str, fps: int = 24, count: int = FIXTURES) -> tuple[int, int]:
    """Write an H.264 preview. Returns (frames, fixture count)."""
    rig = build_rig(img, els, count=count)
    H, W = rig.base.shape[:2]
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path], stdin=subprocess.PIPE)
    n = int(plan.duration_s * fps)
    for i in range(n):
        ff.stdin.write(frame_at(rig, plan, i / fps).clip(0, 255).astype(np.uint8).tobytes())
    ff.stdin.close(); ff.wait()
    return n, rig.n


if __name__ == "__main__":  # python -m app.motion photo.png "brief" out.mp4 [duration_s] [fixture_count]
    import os, pickle
    from app.mapper import map_facade
    from app.planner import make_plan
    im = Image.open(sys.argv[1])
    dur = float(sys.argv[4]) if len(sys.argv) > 4 else 20
    cnt = int(sys.argv[5]) if len(sys.argv) > 5 else FIXTURES
    cache = sys.argv[3] + ".pkl"  # reuse map+plan so motion can be iterated without new model calls

    async def go():
        els = await map_facade(im)
        return els, await make_plan(im, els, sys.argv[2], dur)

    els, plan = pickle.load(open(cache, "rb")) if os.path.exists(cache) else asyncio.run(go())
    pickle.dump((els, plan), open(cache, "wb"))
    for e in els: print(e.id, e.kind, e.label)
    for st in plan.steps: print(st.start_s, st.end_s, st.element_ids, st.motion, st.colors[:2], st.text)
    print("frames, fixtures:", render(im, els, plan, sys.argv[3], count=cnt))
