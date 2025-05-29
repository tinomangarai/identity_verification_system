import streamlit as st
import easyocr
from deepface import DeepFace
import insightface
import cv2
import numpy as np
from PIL import Image
import tempfile

# Load models (cached for performance)
@st.cache_resource
def load_models():
    reader = easyocr.Reader(['en'])
    face_app = insightface.app.FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=-1)  # Use CPU (ctx_id=0 for GPU if supported)
    return reader, face_app

reader, face_app = load_models()

# Title
st.title("🛂 Identity Verification System")

# Upload files
id_image = st.file_uploader("📄 Upload ID Document", type=["jpg", "jpeg", "png"])
selfie_image = st.file_uploader("🤳 Upload Selfie", type=["jpg", "jpeg", "png"])

# Proceed if both are uploaded
if id_image and selfie_image:
    # Convert uploaded files to temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_id, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_selfie:
        tmp_id.write(id_image.getbuffer())
        tmp_selfie.write(selfie_image.getbuffer())
        id_path = tmp_id.name
        selfie_path = tmp_selfie.name

    # OCR
    st.subheader("🔎 OCR Results from ID")
    ocr_results = reader.readtext(id_path)
    if ocr_results:
        for res in ocr_results:
            st.write(f"- {res[1]}")
    else:
        st.warning("No text detected in ID image.")

    # Face Verification
    st.subheader("🧑‍🦱 Face Verification")
    try:
        face_result = DeepFace.verify(img1_path=id_path, img2_path=selfie_path, enforce_detection=False)
        if face_result.get("verified", False):
            st.success("✅ Faces match!")
        else:
            st.error("❌ Faces do not match.")
    except Exception as e:
        st.error(f"Face verification failed: {e}")

    # Liveness Detection
    st.subheader("🧬 Liveness Check")
    try:
        selfie_np = cv2.imread(selfie_path)
        selfie_np = cv2.cvtColor(selfie_np, cv2.COLOR_BGR2RGB)
        faces = face_app.get(selfie_np)

        if faces:
            score = faces[0].det_score
            st.metric("Liveness Score", f"{score:.2f}")
            st.success("✅ Real face detected") if score > 0.5 else st.error("❌ Possible spoof!")
        else:
            st.warning("No face detected for liveness check.")
    except Exception as e:
        st.error(f"Liveness check failed: {e}")
