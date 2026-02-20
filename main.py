import logging
from contextlib import asynccontextmanager

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import insightface
from insightface.app import FaceAnalysis

logger = logging.getLogger("spy-ml")
logging.basicConfig(level=logging.INFO)

face_app: FaceAnalysis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_app
    logger.info("Loading InsightFace model (buffalo_l)...")
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("Model loaded successfully")
    yield
    face_app = None


app = FastAPI(title="SPY ML - Face Embedding Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def download_and_decode_image(image_url: str) -> np.ndarray:
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(image_url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")

    img_array = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


def pick_largest_face(faces: list):
    return sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )[0]


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    image_url: str


class EmbedResponse(BaseModel):
    embedding: list[float]
    face_count: int


class LivenessRequest(BaseModel):
    image_url: str


class LivenessResponse(BaseModel):
    is_live: bool
    score: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/embed", response_model=EmbedResponse)
async def generate_embedding(req: EmbedRequest):
    img = await download_and_decode_image(req.image_url)

    faces = face_app.get(img)
    face_count = len(faces)

    if face_count == 0:
        return EmbedResponse(embedding=[], face_count=0)

    face = pick_largest_face(faces)
    raw = face.embedding
    norm = np.linalg.norm(raw)
    normalized = raw / norm if norm > 0 else raw
    embedding = normalized.tolist()

    return EmbedResponse(embedding=embedding, face_count=face_count)


@app.post("/liveness", response_model=LivenessResponse)
async def check_liveness(req: LivenessRequest):
    img = await download_and_decode_image(req.image_url)

    faces = face_app.get(img)
    if len(faces) == 0:
        return LivenessResponse(is_live=False, score=0.0)

    face = pick_largest_face(faces)

    # Signal 1: Detection confidence (0-1)
    det_score = float(face.det_score)

    # Signal 2: Sharpness via Laplacian variance on face crop
    x1, y1, x2, y2 = [int(c) for c in face.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    face_crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(laplacian_var / 200.0, 1.0)

    # Signal 3: Face size ratio (face area / image area)
    face_area = (x2 - x1) * (y2 - y1)
    img_area = img.shape[0] * img.shape[1]
    size_ratio = min(face_area / img_area / 0.15, 1.0)

    # Weighted composite
    score = round(0.4 * det_score + 0.35 * sharpness + 0.25 * size_ratio, 4)

    return LivenessResponse(is_live=score >= 0.5, score=score)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": face_app is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# cd /Users/chidilonginus/Documents/projects/IcodeIdea/Ai/spy-ml                                                                                                                                                                                                                         
# source venv/bin/activate                                                                                                                                                                                                                                                               
# python main.py
