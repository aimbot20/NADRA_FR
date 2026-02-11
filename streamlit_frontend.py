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

# ======================================
# CAMERA WITH OVERLAY
# ======================================
if input_method == "📸 Use Camera":

    st.markdown("""
    <style>
    .camera-wrapper {
        position: relative;
        display: inline-block;
    }

    .overlay-circle {
        position: absolute;
        left: 50%;
        top: 50%;
        width: 200px;
        height: 250px;
        border-radius: 50%;
        pointer-events: none;
        transform: translate(-59%, -149%);
        border: 2px solid rgba(0, 255, 0, 0.6);
        box-shadow:
            0 0 6px rgba(0, 250, 0, 0.4),
            0 0 15px rgba(0, 250, 0, 0.25),
            inset 0 0 12px rgba(0, 180, 0, 0.2);
        background: radial-gradient(
            ellipse at center,
            rgba(0, 250, 0, 0.05) 0%,
            rgba(0, 250, 0, 0.02) 50%,
            transparent 70%
        );
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="camera-wrapper">', unsafe_allow_html=True)
    uploaded_img = st.camera_input("Take a photo")
    st.markdown('<div class="overlay-circle"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ======================================
# AFTER IMAGE CAPTURE
# ======================================
if uploaded_img is not None:

    image = Image.open(uploaded_img).convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    # ======================================
    # ENROLL
    # ======================================
    if option == "Enroll in Database":

        name = st.text_input("Enter Your Name")

        if st.button("Enroll"):
            if name.strip() == "":
                st.warning("Please enter a name.")
            else:
                files = {"file": ("image.jpg", buf, "image/jpeg")}
                data = {"name": name}

                resp = requests.post(f"{API_URL}/enroll", files=files, data=data)
                result = resp.json()

                if "bbox" in result:
                    x1, y1, x2, y2 = result["bbox"]
                    cropped = image.crop((x1, y1, x2, y2))

                    draw = ImageDraw.Draw(cropped)
                    draw.rectangle(
                        [10, 10, cropped.width - 10, cropped.height - 10],
                        outline="lime",
                        width=4
                    )

                    st.image(cropped, caption="Enrolled Face", use_container_width=False)
                else:
                    st.image(image, caption="Image", use_container_width=False)

                st.json(result)

    # ======================================
    # SEARCH
    # ======================================
    if option == "Find Yourself":

        if st.button("Search"):

            files = {"file": ("image.jpg", buf, "image/jpeg")}
            resp = requests.post(f"{API_URL}/search", files=files)
            result = resp.json()

            if "bbox" in result:
                x1, y1, x2, y2 = result["bbox"]

                cropped = image.crop((x1, y1, x2, y2))

                draw = ImageDraw.Draw(cropped)
                draw.rectangle(
                    [10, 10, cropped.width - 10, cropped.height - 10],
                    outline="lime",
                    width=4
                )

                st.image(cropped, caption="Detected Face", use_container_width=False)
            else:
                st.image(image, caption="Image", use_container_width=False)

            st.json(result)
