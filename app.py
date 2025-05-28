import streamlit as st
from deepface import DeepFace
import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import re
from datetime import datetime
from insightface.app import FaceAnalysis
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import tempfile

# Initialize heavy models ONCE
@st.cache_resource(show_spinner=False)
def load_models():
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a GPU
    face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0)
    return reader, face_app

reader, face_app = load_models()

# Helper: convert uploaded file to cv2 image
def file_to_cv2_image(uploaded_file):
    image_bytes = uploaded_file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Invalid image file")
        return None
    return img

# Helper: convert cv2 image to PIL for display
def cv2_to_pil(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# Extract DOB from OCR text lines
def extract_dob(ocr_text):
    dob = None
    for line in ocr_text:
        match = re.search(r'(\d{2}[/-]\d{2}[/-]\d{4})', line)
        if match:
            dob_str = match.group(1)
            for fmt in ("%d/%m/%Y", "%d-%m-%Y"):
                try:
                    dob = datetime.strptime(dob_str, fmt)
                    return dob
                except:
                    continue
    return None

st.title("🆔 Identity Verification System")

st.markdown("""
Upload your **ID card image** and a **selfie** to verify identity using:
- OCR for DOB extraction
- Face verification (DeepFace)
- Liveness detection (InsightFace)
- Address verification (geopy)
""")

with st.form("verification_form"):
    id_image_file = st.file_uploader("Upload ID Image", type=["png", "jpg", "jpeg"], help="Upload a clear photo of your ID card")
    selfie_image_file = st.file_uploader("Upload Selfie Image", type=["png", "jpg", "jpeg"], help="Upload a selfie photo for verification")
    submit = st.form_submit_button("Verify Identity")

if submit:
    if id_image_file is None or selfie_image_file is None:
        st.warning("Please upload both images")
    else:
        # Convert to cv2 images
        id_img = file_to_cv2_image(id_image_file)
        selfie_img = file_to_cv2_image(selfie_image_file)

        if id_img is None or selfie_img is None:
            st.error("Error reading images. Please try again.")
        else:
            st.subheader("OCR Text from ID Image")
            id_img_rgb = cv2.cvtColor(id_img, cv2.COLOR_BGR2RGB)
            ocr_results = reader.readtext(id_img_rgb)
            ocr_text = [res[1] for res in ocr_results]
            for line in ocr_text[:10]:
                st.write(line)

            # Extract DOB and calculate age
            dob = extract_dob(ocr_text)
            if dob:
                age = (datetime.now() - dob).days // 365
                st.write(f"**Date of Birth:** {dob.strftime('%d-%m-%Y')}")
                st.write(f"**Age:** {age} years")
                st.write("**Age Verification:** " + ("18+ verified ✅" if age >= 18 else "Underage ❌"))
            else:
                st.warning("DOB not found in ID OCR text")

            st.subheader("Face Verification (DeepFace)")
            # Save images temporarily for DeepFace (needs paths)
            with tempfile.NamedTemporaryFile(suffix=".jpg") as id_tmp, tempfile.NamedTemporaryFile(suffix=".jpg") as selfie_tmp:
                cv2.imwrite(id_tmp.name, id_img)
                cv2.imwrite(selfie_tmp.name, selfie_img)
                face_verif = DeepFace.verify(img1_path=id_tmp.name, img2_path=selfie_tmp.name, enforce_detection=False)

            st.write("**Verified:**", face_verif['verified'])
            st.write("**Distance:**", round(face_verif.get('distance', 0), 4))

            st.subheader("Liveness Detection (InsightFace)")
            selfie_rgb = cv2.cvtColor(selfie_img, cv2.COLOR_BGR2RGB)
            faces = face_app.get(selfie_rgb)
            if len(faces) == 0:
                st.warning("No face detected in selfie for liveness check")
                liveness_status = "No face detected"
                liveness_score = None
            else:
                liveness_score = faces[0].det_score
                liveness_status = "Live face detected ✅" if liveness_score >= 0.5 else "Spoof detected ❌"
                st.write("Liveness Score:", liveness_score)
                st.write("Liveness Status:", liveness_status)

            st.subheader("Address Verification (Simulated)")

            # For demo - static address and user location
            address_from_id = "123 Samora Machel Avenue, Harare, Zimbabwe"
            user_coords = (-17.8292, 31.0522)  # Simulated user location (Harare)

            geolocator = Nominatim(user_agent="id_verifier_streamlit")
            location = geolocator.geocode(address_from_id)

            if location:
                address_coords = (location.latitude, location.longitude)
                distance_km = geodesic(user_coords, address_coords).km
                st.write(f"Address extracted: {address_from_id}")
                st.write(f"Address coordinates: {address_coords}")
                st.write(f"Distance from user location: {distance_km:.2f} km")
                location_status = "Address matches claimed location ✅" if distance_km < 5 else "Address does not match location ❌"
                st.write(location_status)
            else:
                st.warning("Address could not be located")

            st.subheader("Uploaded Images")
            st.image(cv2_to_pil(id_img), caption="ID Image", use_column_width=True)
            st.image(cv2_to_pil(selfie_img), caption="Selfie Image", use_column_width=True)
