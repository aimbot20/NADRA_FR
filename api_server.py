
import os
import cv2
import numpy as np
import pickle
import time
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = FastAPI()

# ==========================
# Config
# ==========================
DATABASE_PATH = "database"
EMBED_FILE = "embeddings.pkl"
MODEL_PATH = "model.onnx"
SPOOF_MODEL_PATH = "final_model.h5"
THRESHOLD = 0.5
IMG_SIZE = (224, 224)
SPOOF_MARGIN = 1.5  # margin fraction to expand face bbox

os.makedirs(DATABASE_PATH, exist_ok=True)

# ==========================
# Load Models
# ==========================
print("🔹 Loading models...")
embed_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
embed_input_name = embed_session.get_inputs()[0].name
spoof_model = load_model(SPOOF_MODEL_PATH)
app_detector = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app_detector.prepare(ctx_id=0, det_size=(640, 640))
print("✅ All models loaded successfully.")


# ==========================
# Utility Functions
# ==========================
def detect_spoof_tf(img_rgb):
    img = cv2.resize(img_rgb, IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred_prob = float(spoof_model.predict(img_array, verbose=0)[0][0])
    pred_class = 1 if pred_prob > 0.5 else 0
    label = "SPOOF" if pred_class == 1 else "REAL"
    print(f"[Anti-Spoof] Score={pred_prob:.4f} → {label}")
    return pred_prob, label


def expand_bbox(bbox, image_shape, margin=SPOOF_MARGIN):
    h_img, w_img = image_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    bw, bh = x2 - x1, y2 - y1
    expand_w, expand_h = int(bw * margin), int(bh * margin)
    nx1, ny1 = max(0, x1 - expand_w), max(0, y1 - expand_h)
    nx2, ny2 = min(w_img - 1, x2 + expand_w), min(h_img - 1, y2 + expand_h)
    return (nx1, ny1, nx2, ny2)


def preprocess(img, size=(112, 112)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def get_embedding(img):
    tensor = preprocess(img)
    emb = embed_session.run(None, {embed_input_name: tensor})[0]
    emb = emb.flatten()
    emb = emb / np.linalg.norm(emb)
    return emb


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ==========================
# API Endpoints
# ==========================
@app.post("/enroll")
async def enroll_person(file: UploadFile = File(...), name: str = Form(...)):
    start_total = time.perf_counter()

    img_bytes = await file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Step 1: Face Detection
    t1 = time.perf_counter()
    faces = app_detector.get(frame)
    if len(faces) == 0:
        return JSONResponse({"status": "error", "msg": "No face detected."})
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    x1, y1, x2, y2 = map(int, face.bbox)
    t2 = time.perf_counter()

    # Step 2: Spoof Detection
    sx1, sy1, sx2, sy2 = expand_bbox((x1, y1, x2, y2), frame.shape, margin=SPOOF_MARGIN)
    crop_rgb = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2RGB)
    t_spoof_start = time.perf_counter()
    spoof_score, label = detect_spoof_tf(crop_rgb)
    t_spoof_end = time.perf_counter()
    if label == "SPOOF":
        print(f"⏱ Total={t_spoof_end - start_total:.3f}s (spoof failed)")
        return JSONResponse({
            "status": "error",
            "msg": "❌ Spoof detected. Access denied.",
            "spoof_score": spoof_score
        })

    # Step 3: Embedding Generation
    t_embed_start = time.perf_counter()
    aligned = face_align.norm_crop(frame, face.kps)
    emb = get_embedding(aligned)
    t_embed_end = time.perf_counter()

    # Step 4: Save embedding
    db = {}
    if os.path.exists(EMBED_FILE):
        with open(EMBED_FILE, "rb") as f:
            db = pickle.load(f)
    db[name] = emb
    with open(EMBED_FILE, "wb") as f:
        pickle.dump(db, f)

    t_total = time.perf_counter() - start_total
    print(f"\n🕒 Timing (Enroll: {name})")
    print(f" - Face detection: {t2 - t1:.3f}s")
    print(f" - Spoof detection: {t_spoof_end - t_spoof_start:.3f}s")
    print(f" - Embedding generation: {t_embed_end - t_embed_start:.3f}s")
    print(f" - TOTAL: {t_total:.3f}s\n")

    return JSONResponse({
        "status": "ok",
        "msg": f"{name} enrolled successfully.",
        "spoof_score": spoof_score
    })


@app.post("/search")
async def search_person(file: UploadFile = File(...)):
    start_total = time.perf_counter()

    img_bytes = await file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Step 1: Face Detection
    t1 = time.perf_counter()
    faces = app_detector.get(frame)
    if len(faces) == 0:
        return JSONResponse({"status": "error", "msg": "No face detected."})
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    x1, y1, x2, y2 = map(int, face.bbox)
    t2 = time.perf_counter()

    # Step 2: Spoof Detection
    sx1, sy1, sx2, sy2 = expand_bbox((x1, y1, x2, y2), frame.shape, margin=SPOOF_MARGIN)
    crop_rgb = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2RGB)
    t_spoof_start = time.perf_counter()
    spoof_score, label = detect_spoof_tf(crop_rgb)
    t_spoof_end = time.perf_counter()
    if label == "SPOOF":
        print(f"⏱ Total={t_spoof_end - start_total:.3f}s (spoof failed)")
        return JSONResponse({
            "status": "error",
            "msg": "❌ Spoof detected. Access denied.",
            "spoof_score": spoof_score
        })

    # Step 3: Embedding Generation
    t_embed_start = time.perf_counter()
    aligned = face_align.norm_crop(frame, face.kps)
    emb = get_embedding(aligned)
    t_embed_end = time.perf_counter()

    # Step 4: Search Embeddings
    t_search_start = time.perf_counter()
    if not os.path.exists(EMBED_FILE):
        return JSONResponse({"status": "error", "msg": "No embeddings database found."})
    with open(EMBED_FILE, "rb") as f:
        db = pickle.load(f)

    best_name, best_score = None, -1
    for name, e in db.items():
        score = cosine_similarity(emb, e)
        if score > best_score:
            best_score, best_name = score, name
    t_search_end = time.perf_counter()

    t_total = time.perf_counter() - start_total
    print(f"\n🕒 Timing (Search)")
    print(f" - Face detection: {t2 - t1:.3f}s")
    print(f" - Spoof detection: {t_spoof_end - t_spoof_start:.3f}s")
    print(f" - Embedding generation: {t_embed_end - t_embed_start:.3f}s")
    print(f" - Database search: {t_search_end - t_search_start:.3f}s")
    print(f" - TOTAL: {t_total:.3f}s\n")

    if best_score < THRESHOLD:
        return JSONResponse({
            "status": "ok",
            "msg": "No match found.",
            "score": float(best_score),
            "spoof_score": spoof_score
        })

    return JSONResponse({
        "status": "ok",
        "msg": f"✅ Match found: {best_name}",
        "score": float(best_score),
        "spoof_score": spoof_score
    })






























