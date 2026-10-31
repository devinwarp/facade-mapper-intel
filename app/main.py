from fastapi import FastAPI

app = FastAPI(title="facade-mapper-intel")


@app.get("/health")
def health():
    return {"ok": True}
