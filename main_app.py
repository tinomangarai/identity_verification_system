import streamlit as st
import easyocr
import face_recognition
import numpy as np
import cv2
from PIL import Image
import tempfile

# Load OCR reader
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

st.title("🔐 Lightweight Identity Verification System")

# Upload images
id_image = st.file_uploader("Upload ID Document", type=["jpg", "jpeg", "png"])
selfie_image = st.file_uploader("Upload Selfie", type=["jpg", "jpeg", "png"])

def load_image(file) -> np.ndarray:
    return np.array(Image.open(file).convert('RGB'))

# Proceed only when both files are uploaded
if id_image and selfie_image:
    id_img_np = load_image(id_image)
    selfie_img_np = load_image(selfie_image)

    # OCR on ID image
    st.subheader("🔎 OCR Results")
    with st.spinner("Reading ID document..."):
        ocr_result = reader.readtext(id_img_np)
        if ocr_result:
            for res in ocr_result:
                st.write(f"- {res[1]}")
        else:
            st.warning("No text detected.")

    # Face verification
    st.subheader("🧑‍🦱 Face Verification")
    try:
        id_faces = face_recognition.face_encodings(id_img_np)
        selfie_faces = face_recognition.face_encodings(selfie_img_np)

        if id_faces and selfie_faces:
            match_result = face_recognition.compare_faces([id_faces[0]], selfie_faces[0])[0]
            st.success("✅ Faces match!") if match_result else st.error("❌ Faces don't match.")
        else:
            st.warning("Face not detected in one or both images.")
    except Exception as e:
        st.error(f"Face recognition failed: {e}")

    # Simulated liveness check
    st.subheader("🧬 Simulated Liveness Check")
    try:
        selfie_gray = cv2.cvtColor(selfie_img_np, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(selfie_gray, cv2.CV_64F).var()

        st.metric("Sharpness (Laplacian Variance)", f"{laplacian_var:.2f}")
        if laplacian_var > 100:
            st.success("✅ Likely a real (sharp) photo")
        else:
            st.warning("⚠️ Image may be blurred or spoofed.")
    except Exception as e:
        st.error(f"Liveness check failed: {e}")
