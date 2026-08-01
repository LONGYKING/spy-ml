import asyncio
import base64
import logging
import os
import time
from contextlib import asynccontextmanager

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

import insightface
from insightface.app import FaceAnalysis

logger = logging.getLogger("spy-ml")
logging.basicConfig(level=logging.INFO)

face_app: FaceAnalysis | None = None

# ---------------------------------------------------------------------------
# API key auth
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    expected = os.environ.get("ML_API_KEY", "")
    if not expected:
        # No key configured — allow all (local dev only)
        return ""
    if api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_app
    logger.info("Loading InsightFace model (buffalo_l)...")
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("Model loaded successfully")
    yield
    face_app = None


app = FastAPI(title="Face Discovery — ML Service", lifespan=lifespan)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000
        logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration_ms:.0f}ms)")
        return response


app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _decode_image_bytes(raw: bytes) -> np.ndarray:
    img_array = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="image_decode_failed")
    return img


async def download_and_decode_image(image_ref: str) -> np.ndarray:
    """
    Accepts either a remote URL (http/https) OR inline image bytes as a
    base64 data URI ("data:image/jpeg;base64,...") or a bare base64 string.
    Inline bytes let callers (e.g. the live-scan gateway) skip a media-host
    round trip for transient frames that may never be persisted.
    """
    if image_ref.startswith("data:"):
        # data:<mime>;base64,<payload>
        try:
            _, payload = image_ref.split(",", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid_data_uri")
        try:
            return _decode_image_bytes(base64.b64decode(payload))
        except (ValueError, base64.binascii.Error):
            raise HTTPException(status_code=400, detail="invalid_base64_image")

    if not image_ref.startswith(("http://", "https://")):
        # Bare base64 (no data-URI prefix)
        try:
            return _decode_image_bytes(base64.b64decode(image_ref))
        except (ValueError, base64.binascii.Error):
            raise HTTPException(status_code=400, detail="invalid_base64_image")

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(image_ref)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=400, detail=f"image_download_failed: {e}")

    return _decode_image_bytes(resp.content)


def normalize_embedding(raw: np.ndarray) -> list[float]:
    norm = np.linalg.norm(raw)
    return (raw / norm if norm > 0 else raw).tolist()


def face_to_bbox(face, img: np.ndarray) -> dict:
    x1, y1, x2, y2 = face.bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img.shape[1], int(x2))
    y2 = min(img.shape[0], int(y2))
    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


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


class EmbedManyRequest(BaseModel):
    image_url: str
    min_confidence: float = 0.6


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class FaceEmbedding(BaseModel):
    index: int
    embedding: list[float]
    confidence: float
    bounding_box: BoundingBox
    area: int


class EmbedManyResponse(BaseModel):
    faces: list[FaceEmbedding]
    total_detected: int
    image_width: int
    image_height: int


class LivenessRequest(BaseModel):
    image_url: str


class LivenessResponse(BaseModel):
    is_live: bool
    score: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_api_key)])
async def generate_embedding(req: EmbedRequest):
    """Extract embedding for the single largest face. Used for profile registration."""
    img = await download_and_decode_image(req.image_url)

    loop = asyncio.get_event_loop()
    faces = await loop.run_in_executor(None, face_app.get, img)
    face_count = len(faces)

    if face_count == 0:
        return EmbedResponse(embedding=[], face_count=0)

    face = pick_largest_face(faces)
    return EmbedResponse(
        embedding=normalize_embedding(face.embedding),
        face_count=face_count,
    )


@app.post("/embed-many", response_model=EmbedManyResponse, dependencies=[Depends(require_api_key)])
async def generate_embeddings_many(req: EmbedManyRequest):
    """
    Extract embeddings for ALL detected faces in an image.
    Used for scan/discovery flows — group shots, TV stills, screenshots.
    Returns faces sorted by confidence DESC.
    """
    img = await download_and_decode_image(req.image_url)
    img_h, img_w = img.shape[:2]

    loop = asyncio.get_event_loop()
    all_faces = await loop.run_in_executor(None, face_app.get, img)
    total_detected = len(all_faces)

    # Filter by confidence threshold and sort by confidence DESC
    filtered = sorted(
        [f for f in all_faces if float(f.det_score) >= req.min_confidence],
        key=lambda f: float(f.det_score),
        reverse=True,
    )

    result_faces: list[FaceEmbedding] = []
    for i, face in enumerate(filtered):
        bbox_dict = face_to_bbox(face, img)
        result_faces.append(FaceEmbedding(
            index=i,
            embedding=normalize_embedding(face.embedding),
            confidence=round(float(face.det_score), 4),
            bounding_box=BoundingBox(**bbox_dict),
            area=bbox_dict["width"] * bbox_dict["height"],
        ))

    return EmbedManyResponse(
        faces=result_faces,
        total_detected=total_detected,
        image_width=img_w,
        image_height=img_h,
    )


@app.post("/liveness", response_model=LivenessResponse, dependencies=[Depends(require_api_key)])
async def check_liveness(req: LivenessRequest):
    """Anti-spoofing check for identity verification flow only."""
    img = await download_and_decode_image(req.image_url)

    loop = asyncio.get_event_loop()
    faces = await loop.run_in_executor(None, face_app.get, img)
    if len(faces) == 0:
        return LivenessResponse(is_live=False, score=0.0)

    face = pick_largest_face(faces)

    # Signal 1: Detection confidence (40%)
    det_score = float(face.det_score)

    # Signal 2: Sharpness via Laplacian variance on face crop (35%)
    x1, y1, x2, y2 = [int(c) for c in face.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    face_crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 200.0, 1.0)

    # Signal 3: Face size ratio (25%)
    face_area = (x2 - x1) * (y2 - y1)
    img_area = img.shape[0] * img.shape[1]
    size_ratio = min(face_area / img_area / 0.15, 1.0)

    score = round(0.4 * det_score + 0.35 * sharpness + 0.25 * size_ratio, 4)
    return LivenessResponse(is_live=score >= 0.1, score=score)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": face_app is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# Dev:
# cd /Users/chidilonginus/Documents/projects/IcodeIdea/Ai/spy-ml
# source venv/bin/activate
# python main.py


# Known Limitation (Acceptable for MVP)

#  Liveness is heuristic-based, not a deep-learning anti-spoofing model:
#  The /liveness endpoint computes score = 0.4 * det_confidence + 0.35 * sharpness + 0.25 * size_ratio. A clear printed photo held close to the camera would likely pass. For MVP this is acceptable — noted so the team knows to
#   replace it with a proper anti-spoofing model (e.g. ONNX-based FAS) before produ