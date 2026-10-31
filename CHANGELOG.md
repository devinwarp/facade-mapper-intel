# Changelog

Event work only (26 Oct – 4 Nov 2026). One entry per feature, dated.

## 2026-10-31
- Scaffold: FastAPI service, smoke script for SAM 3.1 / Muse Spark / Muse Image.
- Verified all three models respond from Dubai; SAM box/point prompts work, concept prompts weak on architecture.

## 2026-10-31 (F1)
- F1 read the building: Muse Spark proposes element boxes, SAM 3.1 outlines them; band/fins/glass regions clipped to their parent. `POST /map` and `python -m app.mapper`. First test on a night photo of two curved towers: 7 elements in ~18 s, no hand-drawn zones.

## 2026-10-31 (F2)
- F2 show plan: Muse Spark writes a timed plan (JSON schema); every step must name real element ids, invalid plans are re-asked once. `POST /plan`, `python -m app.planner`.
