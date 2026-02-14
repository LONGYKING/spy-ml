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


class EmbedRequest(BaseModel):
    image_url: str


class EmbedResponse(BaseModel):
    embedding: list[float]
    face_count: int


@app.post("/embed", response_model=EmbedResponse)
async def generate_embedding(req: EmbedRequest):
    # Download image from URL
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(req.image_url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")

    # Decode image
    img_array = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Detect faces and extract embeddings
    faces = face_app.get(img)
    face_count = len(faces)

    if face_count == 0:
        return EmbedResponse(embedding=[], face_count=0)

    # Return the first (largest) face's embedding
    # Sort by bounding box area descending to pick the most prominent face
    faces_sorted = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    raw = faces_sorted[0].embedding
    norm = np.linalg.norm(raw)
    normalized = raw / norm if norm > 0 else raw
    embedding = normalized.tolist()

    return EmbedResponse(embedding=embedding, face_count=face_count)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": face_app is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
