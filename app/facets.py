"""Facets: which survey elevation lands on which visible face of the building, and where.

A survey holds one canvas region per face (an elevation). Each face visible in the photo needs its own quad;
faces turned away contribute nothing. Muse Spark proposes the visible faces' corners, and the assignment of
elevations to faces is chosen by how well the mapped fixtures fit the tower's silhouette."""
import itertools, json

import numpy as np
from PIL import Image, ImageFilter

from app import survey as sv
from app.mapper import _client, _data_url

FACE_SCHEMA = {
    "type": "object",
    "properties": {"faces": {"type": "array", "items": {
        "type": "object",
        "properties": {k: {"type": "integer"} for k in ("tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y")},
        "required": ["tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"],
        "additionalProperties": False,
    }}},
    "required": ["faces"],
    "additionalProperties": False,
}

PROMPT = (
    "This photo shows a tall building with several faces around it. List every face of the main tower that is "
    "visible, left to right (usually 1 to 3). Neighbouring faces meet at a vertical corner line and share that "
    "edge. For each face give the outline's four corners as top-left, top-right, bottom-right, bottom-left, "
    "normalised to 0-1000 of the image width and height. Cover the tower only, not the low podium, annex, sky "
    "or foreground. Where the tower twists, give the best straight-edged quad."
)


async def propose_faces(img: Image.Image) -> list[np.ndarray]:
    """Quads (4x2, photo pixels, TL TR BR BL), left to right."""
    r = await _client().chat.completions.create(
        model="muse-spark-1.3",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": _data_url(img)}},
        ]}],
        response_format={"type": "json_schema", "json_schema": {"name": "faces", "schema": FACE_SCHEMA, "strict": True}},
    )
    w, h = img.size
    quads = []
    for f in json.loads(r.choices[0].message.content)["faces"]:
        q = np.array([[f[f"{c}_x"] * w / 1000, f[f"{c}_y"] * h / 1000] for c in ("tl", "tr", "br", "bl")], float)
        quads.append(q)
    return sorted(quads, key=lambda q: q[:, 0].mean())


def fit_score(survey: sv.Survey, pairs, tower: np.ndarray) -> float:
    """How well mapped fixtures cover the tower silhouette without spilling off it (0..1)."""
    H, W = tower.shape
    cov = np.zeros((H, W), bool); spill = 0; total = 0
    for rect, quad in pairs:
        X, Y = sv.map_elevation(survey, rect, quad)
        xi, yi = np.clip(X.astype(int), 0, W - 1), np.clip(Y.astype(int), 0, H - 1)
        spill += (~tower[yi, xi]).sum(); total += len(X)
        cov[yi, xi] = True
    foot = np.asarray(Image.fromarray(cov.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(15))) > 0
    coverage = (foot & tower).sum() / max(tower.sum(), 1)
    return float(coverage - spill / max(total, 1))


def assign(survey: sv.Survey, quads: list[np.ndarray], tower: np.ndarray):
    """Choose which consecutive elevations (cyclic, in survey order) the visible faces are.
    Returns [(canvas rect, quad)] and the winning score."""
    rects = sv.elevations(survey)
    k = len(quads)
    best = (-9.0, None)
    for start in range(len(rects)):
        pairs = [(rects[(start + i) % len(rects)], quads[i]) for i in range(k)]
        s = fit_score(survey, pairs, tower)
        if s > best[0]:
            best = (s, pairs)
    return best[1], best[0]


def silhouette_edges(tower: np.ndarray, trim: float = 0.08):
    """Left and right edges of the tower as straight lines fitted over its height (twist makes them approximate).
    Returns (y_top, y_bottom, left(y), right(y))."""
    ys = np.where(tower.any(axis=1))[0]
    y0, y1 = ys.min(), ys.max()
    rows = ys[(ys >= y0 + trim * (y1 - y0)) & (ys <= y1 - trim * (y1 - y0))]
    left = np.array([np.where(tower[y])[0].min() for y in rows])
    right = np.array([np.where(tower[y])[0].max() for y in rows])
    fl, fr = np.polyfit(rows, left, 1), np.polyfit(rows, right, 1)
    return float(y0), float(y1), (lambda y: np.polyval(fl, y)), (lambda y: np.polyval(fr, y))


def snap_to_silhouette(quads: list[np.ndarray], tower: np.ndarray) -> list[np.ndarray]:
    """Spark's outer corners are loose; SAM's tower silhouette is exact. Pin the outer edges to it and
    keep neighbouring faces sharing one edge (a building corner)."""
    yt, yb, L, R = silhouette_edges(tower)
    q = [x.copy() for x in quads]
    q[0][0], q[0][3] = (L(yt), yt), (L(yb), yb)
    q[-1][1], q[-1][2] = (R(yt), yt), (R(yb), yb)
    for a, b in zip(q, q[1:]):  # shared edge: right edge of a == left edge of b
        top, bot = (a[1] + b[0]) / 2, (a[2] + b[3]) / 2
        a[1] = b[0] = top
        a[2] = b[3] = bot
    for f in q:  # one shared top and bottom line
        f[0, 1] = f[1, 1] = yt
        f[2, 1] = f[3, 1] = yb
    return q


async def map_faces(img: Image.Image, survey: sv.Survey):
    """Photo + survey -> (tower silhouette, [(canvas rect, photo quad)] per visible face)."""
    from app.mapper import tower_mask
    tower = await tower_mask(img)
    pairs, _ = fit_faces(survey, await propose_faces(img), tower)
    return tower, pairs


def fit_faces(survey: sv.Survey, quads: list[np.ndarray], tower: np.ndarray, steps: int = 9):
    """Two visible faces: Spark's shared building-corner edge is unstable, so search it (top and bottom x) for
    the position where mapped fixtures best fit the SAM silhouette. Other counts: snap Spark's quads.
    ponytail: straight edges; a twisting corner needs a curve or more facets."""
    if len(quads) != 2:
        return assign(survey, snap_to_silhouette(quads, tower), tower)
    yt, yb, L, R = silhouette_edges(tower)
    span = lambda y: np.linspace(L(y) + 0.15 * (R(y) - L(y)), R(y) - 0.15 * (R(y) - L(y)), steps)
    best = (-9.0, None)
    for xt in span(yt):
        for xb in span(yb):
            q1 = np.array([[L(yt), yt], [xt, yt], [xb, yb], [L(yb), yb]], float)
            q2 = np.array([[xt, yt], [R(yt), yt], [R(yb), yb], [xb, yb]], float)
            pairs, s = assign(survey, [q1, q2], tower)
            if s > best[0]:
                best = (s, pairs)
    return best[1], best[0]
