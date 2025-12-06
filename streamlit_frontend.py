
import streamlit as st
import requests
import io
from PIL import Image

API_URL = "http://localhost:8000"  # FastAPI backend address

st.title("Face Recognition Portal")
st.subheader("Enroll or Identify Yourself")

option = st.radio("Choose Action", ["Enroll in Database", "Find Yourself"])

# Camera or file upload choice
input_method = st.radio("Select Input Method", ["📸 Use Camera", "📁 Upload Image"])

uploaded_img = None
if input_method == "📸 Use Camera":
    uploaded_img = st.camera_input("Take a photo")
else:
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    image = Image.open(uploaded_img)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    if option == "Enroll in Database":
        name = st.text_input("Enter your name")
        if st.button("Enroll"):
            if not name.strip():
                st.warning("Please enter a name.")
            else:
                files = {"file": ("image.jpg", buf, "image/jpeg")}
                data = {"name": name}
                resp = requests.post(f"{API_URL}/enroll", files=files, data=data)
                st.json(resp.json())

    elif option == "Find Yourself":
        if st.button("Search"):
            files = {"file": ("image.jpg", buf, "image/jpeg")}
            resp = requests.post(f"{API_URL}/search", files=files)
            st.json(resp.json())





