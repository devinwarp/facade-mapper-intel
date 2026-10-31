# facade-mapper-intel

Reads a building's architecture from one photo and makes show content that fits it.

- **SAM 3.1** maps the facade's elements (fins, glass bands, setbacks, crown).
- **Muse Spark** plans the show against those elements and checks readability.
- **Muse Image** makes per-element visuals.
- Our own code moves content along element outlines.

## Built during the event

Everything in this repository was written 26 Oct – 4 Nov 2026 for the Meta AI Developer Hackathon (Startup track).
Commit history is the evidence; see [CHANGELOG.md](CHANGELOG.md).

> A pre-existing Facade Mapper app exists separately (private) and is **not** submitted.
> This repo runs as a standalone service with its own interface; the existing app calls it.

## Run

```
cp .env.example .env   # add MODEL_API_KEY from https://dev.meta.ai
uv sync
uv run python scripts/smoke.py [facade.jpg]
uv run uvicorn app.main:app --reload
uv run pytest
```

Hosting: Render web service (long-running Python process; no function timeouts for model calls).

License: Apache-2.0
