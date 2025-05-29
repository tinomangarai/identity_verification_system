import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Load OCR
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

st.title("🔐 Lightweight Identity Verification System")

# Upload images
id_file = st.file_uploader("Upload ID Document", type=["jpg", "jpeg", "png"])
selfie_file = st.file_uploader("Upload Selfie", type=["jpg", "jpeg", "png"])

# Load images
def read_image(file):
    return np.array(Image.open(file).convert('RGB'))

# Detect face using OpenCV's DNN face detector
def detect_face(image_np):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return image_np[y:y+h, x:x+w]

# Compare two face crops using cosine similarity
def compare_faces(face1, face2):
    try:
        face1_resized = cv2.resize(face1, (100, 100)).flatten().reshape(1, -1)
        face2_resized = cv2.resize(face2, (100, 100)).flatten().reshape(1, -1)
        similarity = cosine_similarity(face1_resized, face2_resized)[0][0]
        return similarity
    except Exception as e:
        return None

if id_file and selfie_file:
    id_img = read_image(id_file)
    selfie_img = read_image(selfie_file)

    st.subheader("🔎 OCR from ID Document")
    with st.spinner("Extracting text..."):
        ocr_results = reader.readtext(id_img)
        for result in ocr_results:
            st.write(f"- {result[1]}")

    st.subheader("🧑‍🦱 Face Detection and Verification")
    face_id = detect_face(id_img)
    face_selfie = detect_face(selfie_img)

    if face_id is None or face_selfie is None:
        st.warning("Could not detect face in one of the images.")
    else:
        st.image([face_id, face_selfie], caption=["ID Face", "Selfie Face"], width=150)

        score = compare_faces(face_id, face_selfie)
        if score is not None:
            st.metric("Cosine Similarity", f"{score:.2f}")
            if score > 0.8:
                st.success("✅ Faces likely match.")
            else:
                st.error("❌ Faces do not match.")
        else:
            st.error("Face comparison failed.")

    st.subheader("🧬 Simulated Liveness Check")
    sharpness = cv2.Laplacian(cv2.cvtColor(selfie_img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    st.metric("Sharpness (Laplacian)", f"{sharpness:.2f}")
    if sharpness > 100:
        st.success("✅ Image is sharp — likely a live photo.")
    else:
        st.warning("⚠️ Image is blurry — may be spoofed.")
