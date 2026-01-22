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
SPOOF_MARGIN = 2

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

# 🔹 ADDED: unified face selection
def get_largest_face(frame):
    faces = app_detector.get(frame)
    if len(faces) == 0:
        return None, None
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    x1, y1, x2, y2 = map(int, face.bbox)
    return face, (x1, y1, x2, y2)


def detect_spoof_tf(img_rgb):
    img = cv2.resize(img_rgb, IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred_prob = float(spoof_model.predict(img_array, verbose=0)[0][0])
    label = "SPOOF" if pred_prob > 0.5 else "REAL"
    return pred_prob, label


def expand_bbox(bbox, image_shape, margin=SPOOF_MARGIN):
    h_img, w_img = image_shape[:2]
    x1, y1, x2, y2 = bbox
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
    return emb / np.linalg.norm(emb)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ==========================
# API Endpoints
# ==========================
@app.post("/enroll")
async def enroll_person(file: UploadFile = File(...), name: str = Form(...)):
    img_bytes = await file.read()
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 🔹 MODIFIED: face detection
    face, bbox = get_largest_face(frame)
    if face is None:
        return JSONResponse({"status": "error", "msg": "No face detected."})

    x1, y1, x2, y2 = bbox

    # Spoof detection
    sx1, sy1, sx2, sy2 = expand_bbox(bbox, frame.shape)
    crop_rgb = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2RGB)
    spoof_score, label = detect_spoof_tf(crop_rgb)
    if label == "SPOOF":
        return JSONResponse({
            "status": "error",
            "msg": "❌ Spoof detected. Access denied.",
            "spoof_score": spoof_score
        })

    # Embedding
    aligned = face_align.norm_crop(frame, face.kps)
    emb = get_embedding(aligned)

    db = {}
    if os.path.exists(EMBED_FILE):
        with open(EMBED_FILE, "rb") as f:
            db = pickle.load(f)
    db[name] = emb
    with open(EMBED_FILE, "wb") as f:
        pickle.dump(db, f)

    return JSONResponse({
        "status": "ok",
        "msg": f"{name} enrolled successfully.",
        "spoof_score": spoof_score,
        "bbox": [x1, y1, x2, y2]  # 🔹 ADDED
    })


@app.post("/search")
async def search_person(file: UploadFile = File(...)):
    img_bytes = await file.read()
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 🔹 MODIFIED: face detection
    face, bbox = get_largest_face(frame)
    if face is None:
        return JSONResponse({"status": "error", "msg": "No face detected."})

    x1, y1, x2, y2 = bbox

    # Spoof detection
    sx1, sy1, sx2, sy2 = expand_bbox(bbox, frame.shape)
    crop_rgb = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2RGB)
    spoof_score, label = detect_spoof_tf(crop_rgb)
    if label == "SPOOF":
        return JSONResponse({
            "status": "error",
            "msg": "❌ Spoof detected. Access denied.",
            "spoof_score": spoof_score
        })

    # Embedding
    aligned = face_align.norm_crop(frame, face.kps)
    emb = get_embedding(aligned)

    if not os.path.exists(EMBED_FILE):
        return JSONResponse({"status": "error", "msg": "No embeddings database found."})

    with open(EMBED_FILE, "rb") as f:
        db = pickle.load(f)

    best_name, best_score = None, -1
    for name, e in db.items():
        score = cosine_similarity(emb, e)
        if score > best_score:
            best_score, best_name = score, name

    if best_score < THRESHOLD:
        return JSONResponse({
            "status": "ok",
            "msg": "No match found.",
            "score": float(best_score),
            "spoof_score": spoof_score,
            "bbox": [x1, y1, x2, y2]  # 🔹 ADDED
        })

    return JSONResponse({
        "status": "ok",
        "msg": f"✅ Match found: {best_name}",
        "score": float(best_score),
        "spoof_score": spoof_score,
        "bbox": [x1, y1, x2, y2]  # 🔹 ADDED
    })


