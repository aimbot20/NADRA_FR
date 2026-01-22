
### add bounding box 
import streamlit as st
import requests
import io
from PIL import Image, ImageDraw

API_URL = "http://localhost:8000"

st.title("Face Recognition Portal")
st.subheader("Enroll or Identify Yourself")

option = st.radio("Choose Action", ["Enroll in Database", "Find Yourself"])
input_method = st.radio("Select Input Method", ["📸 Use Camera", "📁 Upload Image"])

uploaded_img = None
if input_method == "📸 Use Camera":
    uploaded_img = st.camera_input("Take a photo")
else:
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    image = Image.open(uploaded_img).convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    def draw_and_crop(img, bbox, scale=2.0):
        x1, y1, x2, y2 = bbox
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=4)

        # Face width and height
        face_w = x2 - x1
        face_h = y2 - y1

        # Face center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Crop boundaries with 2x scale
        crop_x1 = max(0, int(cx - face_w * scale / 2))
        crop_x2 = min(img.width, int(cx + face_w * scale / 2))
        crop_y1 = max(0, int(cy - face_h * scale / 2))
        crop_y2 = min(img.height, int(cy + face_h * scale / 2))

        return img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    if option == "Enroll in Database":
        name = st.text_input("Enter your name")
        if st.button("Enroll"):
            if not name.strip():
                st.warning("Please enter a name.")
            else:
                files = {"file": ("image.jpg", buf, "image/jpeg")}
                data = {"name": name}
                resp = requests.post(f"{API_URL}/enroll", files=files, data=data)
                result = resp.json()

                if "bbox" in result:
                    image = draw_and_crop(image, result["bbox"], scale=2.0)

                st.image(image, caption="Detected Face", use_container_width=False)
                st.json(result)

    elif option == "Find Yourself":
        if st.button("Search"):
            files = {"file": ("image.jpg", buf, "image/jpeg")}
            resp = requests.post(f"{API_URL}/search", files=files)
            result = resp.json()

            if "bbox" in result:
                image = draw_and_crop(image, result["bbox"], scale=2.0)

            st.image(image, caption="Detected Face", use_container_width=False)
            st.json(result)
