# NLA Demo Server

FastAPI server that loads the trained NLA (target model `T`, Verbalizer `AV`,
Reconstructor `AR`) and exposes a small JSON API plus the static frontend.

## Prerequisites

- The trained checkpoints in place:
  - `models/av.pt`
  - `models/ar.pt`
- A GPU available (`CUDA_VISIBLE_DEVICES` set as needed).
- Python deps installed (`pip install -r requirements.txt`) — these now include
  `fastapi` and `uvicorn`.

## Run

From the repo root:

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000/> in a browser.

The first request takes a few seconds while the models load; check
`GET /api/health` to confirm `"status": "ok"`.

## API

- `GET /api/health` — sanity check; reports checkpoint paths.
- `POST /api/tokenize` — body `{ "text": "..." }` → `{ "tokens": [...] }`.
- `POST /api/analyze` — body `{ "text": "...", "position": int }` →
  `{ "tokens": [...], "position": int, "explanation": str, "reconstruction": float }`.
  `position` is the token index (negative values count from the end);
  `reconstruction` is the cosine between the AR's prediction and the
  sqrt(d_model)-normalised activation.

FastAPI's auto-generated docs are at <http://localhost:8000/docs>.
