# working with 2 spoofing models implemented (UPDATED FIRST MODEL)
import time
import os
import cv2
import numpy as np
import pickle
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

app = FastAPI()

# ==========================
# Config
# ==========================
DATABASE_PATH = "database"
EMBED_FILE = "embeddings.pkl"
MODEL_PATH = "model.onnx"
# first spoofing model 
SPOOF_MODEL_PATH = "final_model.pth"
# 2nd spoofing model to idnetify type of spoofing 
ATTACK_MODEL_PATH = "best_finetuned.pth"

THRESHOLD = 0.5
IMG_SIZE = 224
SPOOF_MARGIN = 1   # 🔹 Increase or decrease background area here

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(DATABASE_PATH, exist_ok=True)

# ==========================
# Load Models
# ==========================
print("🔹 Loading models...")

embed_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
embed_input_name = embed_session.get_inputs()[0].name

# ==========================
# 1️⃣ Binary Spoof Model
# ==========================
spoof_model = models.densenet121(pretrained=False)

spoof_model.classifier = nn.Sequential(
    nn.Linear(spoof_model.classifier.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 1)
)

spoof_model.load_state_dict(torch.load(SPOOF_MODEL_PATH, map_location=DEVICE))
spoof_model = spoof_model.to(DEVICE)
spoof_model.eval()

spoof_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# 2️⃣ Attack Type Model
# ==========================
idx_to_class = {
    1: "Photo",
    2: "Poster",
    3: "A4",
    4: "Face Mask",
    5: "Upper Body Mask",
    7: "PC",
    8: "Pad",
    9: "Phone"
}

num_classes = len(idx_to_class)

attack_model = models.densenet121(pretrained=False)
num_ftrs = attack_model.classifier.in_features
attack_model.classifier = torch.nn.Linear(num_ftrs, num_classes)

attack_model.load_state_dict(torch.load(ATTACK_MODEL_PATH, map_location=DEVICE))
attack_model = attack_model.to(DEVICE)
attack_model.eval()

attack_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Face detector
app_detector = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app_detector.prepare(ctx_id=0, det_size=(640, 640))

print("✅ All models loaded successfully.")

# ==========================
# Utility Functions
# ==========================

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


def expand_bbox(bbox, image_shape, margin=SPOOF_MARGIN):
    h_img, w_img = image_shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    expand_w = int(bw * margin)
    expand_h = int(bh * margin)

    nx1 = max(0, x1 - expand_w)
    ny1 = max(0, y1 - expand_h)
    nx2 = min(w_img - 1, x2 + expand_w)
    ny2 = min(h_img - 1, y2 + expand_h)

    return nx1, ny1, nx2, ny2


def detect_spoof_torch(img_rgb):
    pil_img = Image.fromarray(img_rgb)
    tensor = spoof_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = spoof_model(tensor)
        prob = torch.sigmoid(output).item()

    label = "SPOOF" if prob > THRESHOLD else "REAL"
    return prob, label


def detect_attack_type(img_rgb):
    pil_img = Image.fromarray(img_rgb)
    tensor = attack_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = attack_model(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    attack_type = idx_to_class.get(pred_idx, "Unknown")
    return attack_type, round(confidence, 4)


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
# SEARCH ENDPOINT
# ==========================
@app.post("/search")
async def search_person(file: UploadFile = File(...)):

    start_time = time.time()  # ⏱ start timer

    img_bytes = await file.read()
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    face, bbox = get_largest_face(frame)
    if face is None:
        return JSONResponse({
            "status": "error",
            "msg": "No face detected."
        })

    x1, y1, x2, y2 = bbox

    sx1, sy1, sx2, sy2 = expand_bbox(bbox, frame.shape)
    crop_rgb = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2RGB)

    spoof_score, label = detect_spoof_torch(crop_rgb)

    if label == "SPOOF":
        attack_type, attack_conf = detect_attack_type(crop_rgb)
        return JSONResponse({
            "status": "error",
            "msg": "❌ Spoof detected. Access denied.",
            "spoof_score": round(spoof_score, 4),
            "attack_type": attack_type,
            "attack_confidence": attack_conf
        })

    aligned = face_align.norm_crop(frame, face.kps)
    emb = get_embedding(aligned)

    if not os.path.exists(EMBED_FILE):
        return JSONResponse({
            "status": "error",
            "msg": "No embeddings database found."
        })

    with open(EMBED_FILE, "rb") as f:
        db = pickle.load(f)

    best_name, best_score = None, -1
    for name, e in db.items():
        score = cosine_similarity(emb, e)
        if score > best_score:
            best_score, best_name = score, name

    end_time = time.time()  # ⏱ end timer
    total_time = round(end_time - start_time, 4)

    response_data = {
        "status": "ok",
        "Matching score": round(float(best_score), 4),
        "spoof_score": round(spoof_score, 4),
        "spoof_msg": "No spoof detected.",
        "processing_time": total_time,  # ⏱ added
        "bbox": [int(x1), int(y1), int(x2), int(y2)]
    }

    if best_score < THRESHOLD:
        response_data["msg"] = "No match found."
    else:
        response_data["msg"] = f"✅ Match found: {best_name}"

    return JSONResponse(response_data)

# ==========================
# ENROLL ENDPOINT
# ==========================
@app.post("/enroll")
async def enroll_person(file: UploadFile = File(...), name: str = Form(...)):

    img_bytes = await file.read()
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Detect largest face
    face, bbox = get_largest_face(frame)
    if face is None:
        return JSONResponse({"status": "error", "msg": "No face detected."})

    x1, y1, x2, y2 = bbox  # tight bbox

    # 🔹 Expand bbox for spoof detection
    sx1, sy1, sx2, sy2 = expand_bbox(bbox, frame.shape)
    crop_rgb = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2RGB)

    # 🔹 Run binary spoof detection
    spoof_score, label = detect_spoof_torch(crop_rgb)

    if label == "SPOOF":
        attack_type, attack_conf = detect_attack_type(crop_rgb)
        return JSONResponse({
            "status": "error",
            "msg": "❌ Spoof detected. Enrollment denied.",
            "spoof_score": round(spoof_score, 4),
            "attack_type": attack_type,
            "attack_confidence": attack_conf
        })

    # REAL → continue enrollment
    aligned = face_align.norm_crop(frame, face.kps)
    emb = get_embedding(aligned)

    # Load existing DB
    db = {}
    if os.path.exists(EMBED_FILE):
        with open(EMBED_FILE, "rb") as f:
            db = pickle.load(f)

    # Save / overwrite embedding
    db[name] = emb

    with open(EMBED_FILE, "wb") as f:
        pickle.dump(db, f)

    return JSONResponse({
        "status": "ok",
        "msg": f"✅ {name} enrolled successfully.",
        "spoof_score": round(spoof_score, 4),
        "spoof_msg": "No spoof detected.",
        "bbox": [int(x1), int(y1), int(x2), int(y2)]
    })


# ==========================
# BULK ENROLL DIRECTORY
# ==========================
@app.post("/bulk_enroll_directory")
async def bulk_enroll_directory(directory_path: str = Form(...)):

    if not os.path.exists(directory_path):
        return JSONResponse({
            "status": "error",
            "msg": "Directory does not exist."
        })

    image_extensions = [".jpg", ".jpeg", ".png"]
    files = [f for f in os.listdir(directory_path)
             if os.path.splitext(f)[1].lower() in image_extensions]

    if len(files) == 0:
        return JSONResponse({
            "status": "error",
            "msg": "No valid images found in directory."
        })

    # Load existing DB
    db = {}
    if os.path.exists(EMBED_FILE):
        with open(EMBED_FILE, "rb") as f:
            db = pickle.load(f)

    enrolled_count = 0
    skipped_spoof = 0
    skipped_no_face = 0

    for file_name in files:

        image_path = os.path.join(directory_path, file_name)
        frame = cv2.imread(image_path)

        if frame is None:
            continue

        face, bbox = get_largest_face(frame)
        if face is None:
            skipped_no_face += 1
            continue

        # Expand bbox for spoof detection
        sx1, sy1, sx2, sy2 = expand_bbox(bbox, frame.shape)
        crop_rgb = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2RGB)

        spoof_score, label = detect_spoof_torch(crop_rgb)

        if label == "SPOOF":
            skipped_spoof += 1
            continue  # 🚫 Ignore spoof images

        # REAL → extract embedding
        aligned = face_align.norm_crop(frame, face.kps)
        emb = get_embedding(aligned)

        # Use filename (without extension) as identity
        identity_name = os.path.splitext(file_name)[0]
        db[identity_name] = emb

        enrolled_count += 1

    # Save updated DB
    with open(EMBED_FILE, "wb") as f:
        pickle.dump(db, f)

    return JSONResponse({
        "status": "ok",
        "msg": "Bulk enrollment completed.",
        "total_images": len(files),
        "enrolled": enrolled_count,
        "skipped_spoof": skipped_spoof,
        "skipped_no_face": skipped_no_face
    })
