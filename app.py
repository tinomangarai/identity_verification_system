import streamlit as st
import easyocr, deepface, insightface, cv2, re
from datetime import datetime
from PIL import Image
import os

# Initialize models once
@st.cache_resource
def load_models():
    reader = easyocr.Reader(['en'])
    face_app = insightface.app.FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=0)
    return reader, face_app

reader, face_app = load_models()

# Streamlit UI
st.title("Identity Verification System")

# File uploaders
id_image = st.file_uploader("Upload ID Document", type=["jpg", "png"])
selfie_image = st.file_uploader("Upload Selfie", type=["jpg", "png"])

if id_image and selfie_image:
    # Save uploaded files
    with open("temp_id.jpg", "wb") as f:
        f.write(id_image.getbuffer())
    with open("temp_selfie.jpg", "wb") as f:
        f.write(selfie_image.getbuffer())

    # OCR Processing
    st.subheader("OCR Results")
    ocr_results = reader.readtext("temp_id.jpg")
    id_text = [r[1] for r in ocr_results]
    for line in id_text:
        st.write(f"- {line}")

    # Face Verification
    st.subheader("Face Verification")
    face_result = deepface.verify("temp_id.jpg", "temp_selfie.jpg")
    st.success("✅ Faces match!") if face_result["verified"] else st.error("❌ Faces don't match!")

    # Liveness Detection
    st.subheader("Liveness Check")
    img = cv2.imread("temp_selfie.jpg")
    faces = face_app.get(img)
    if faces:
        liveness_score = faces[0].det_score
        st.metric("Liveness Score", f"{liveness_score:.2f}")
        st.success("✅ Real face detected") if liveness_score >= 0.5 else st.error("❌ Possible spoof!")
    else:
        st.warning("No face detected")

    # Cleanup
    os.remove("temp_id.jpg")
    os.remove("temp_selfie.jpg")
