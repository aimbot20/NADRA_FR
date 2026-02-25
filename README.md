# Face Recognition 

---
# **Setup Instructions: **

requirement:  python 3.9 


pip install -r requirements.txt 


---

place spoofing and embeding models in the same folder 


best_finetuned.pth, 
final_model.pth, 
model.onnx


---



in terminal 1 run 

uvicorn api_server:app --reload --port 8000


terminal 2 run  

streamlit run streamlit_frontend.py


---

app available on 
  Local URL: http://localhost:8501
  
  Network URL: http://192.168.1.14:8501





---------------------------------------
--------------------------------------
---------------------------------------
--------------------------------------
# Results Interpretation: 

Field Descriptions

1️⃣ status

Indicates the overall request result.

Possible values:

"ok" → Request processed successfully

"error" → Something failed (e.g., no face detected, spoof detected, no embeddings in database)

----------
2️⃣ Matching score

Cosine similarity between the query face and the best match in the database.

Range: 0.0 (completely different) → 1.0 (identical)

Threshold = 0.5 :

if value > threshold → Strong match

if value < threshold → No match found

-------
3️⃣ spoof_score

Probability that the detected face is fake or a spoof attack.

Range: 0.0 (real) → 1.0 (definitely spoof)

Interpretation:

if value < 0.5 → Real face

if value >= 0.5 → Spoof detected

----------
5️⃣ attack_type (only present if spoof detected)

Type of spoof detected.

Possible values:

"Photo", "Poster", "A4", "Face Mask", "Upper Body Mask", "PC", "Pad", "Phone"

----------------------

6️⃣ attack_confidence (only present if spoof detected)

Confidence of the predicted spoof type.

Range: 0.0 → 1.0

Higher values indicate stronger certainty of the spoof type.

------------

4️⃣ spoof_msg

Human-readable description of spoof detection.

Possible values:

"No spoof detected." → Liveness verified

"❌ Spoof detected. Access denied." → Spoof detected, access blocked

----------------
5️⃣ processing_time

Total time in seconds to process the search request.

Includes: face detection, spoof check, alignment, embedding generation, and database matching

----------------
6️⃣ bbox

Coordinates of the detected face in the original image.

Format: [x1, y1, x2, y2]

(x1, y1) → Top-left corner

(x2, y2) → Bottom-right corner

Useful for:

Cropping the face

Drawing detection rectangle

Aligning the face before embedding

-------------------
7️⃣ msg

User-facing message about the recognition result.

Possible values:

"✅ Match found: <Name>" → Face recognized successfully

"No match found." → No matching face in the database

"❌ Spoof detected. Access denied." → Spoof attack detected
