import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, date
import pytz
import requests
import re
from dateutil.relativedelta import relativedelta

# Load OCR
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

st.title("🔐 Enhanced Identity Verification System")

# User information form
with st.expander("🔍 Enter Your Information"):
    user_dob = st.date_input("Your Date of Birth", min_value=date(1900, 1, 1))
    user_address = st.text_input("Your Current Address")
    user_country = st.text_input("Country")
    user_state = st.text_input("State/Province")
    user_city = st.text_input("City")
    user_postal = st.text_input("Postal/Zip Code")

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

# Extract date of birth from OCR text
def extract_dob(text_list):
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
        r'\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},\s\d{2,4}\b'
    ]
    
    for text in text_list:
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    parsed_date = datetime.strptime(matches[0], "%m/%d/%Y").date()
                    return parsed_date
                except:
                    try:
                        parsed_date = datetime.strptime(matches[0], "%d-%m-%Y").date()
                        return parsed_date
                    except:
                        pass
    return None

# Calculate age from date of birth
def calculate_age(dob):
    today = date.today()
    age = relativedelta(today, dob).years
    return age

# Verify address using Nominatim (OpenStreetMap)
def verify_address(address, country, state, city, postal_code):
    try:
        query = {
            'street': address,
            'country': country,
            'state': state,
            'city': city,
            'postalcode': postal_code,
            'format': 'json',
            'limit': 1
        }
        
        # Remove None values
        query = {k: v for k, v in query.items() if v is not None}
        
        base_url = "https://nominatim.openstreetmap.org/search"
        headers = {'User-Agent': 'IDVerificationSystem/1.0'}
        
        response = requests.get(base_url, params=query, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        if data:
            return True, data[0]['display_name']
        return False, "Address not found"
    except Exception as e:
        return False, f"Error verifying address: {str(e)}"

if id_file and selfie_file:
    id_img = read_image(id_file)
    selfie_img = read_image(selfie_file)

    st.subheader("🔎 OCR from ID Document")
    with st.spinner("Extracting text..."):
        ocr_results = reader.readtext(id_img)
        extracted_text = [result[1] for result in ocr_results]
        for text in extracted_text:
            st.write(f"- {text}")

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

    st.subheader("📅 Age Verification")
    dob_from_id = extract_dob(extracted_text)
    
    if dob_from_id:
        st.write(f"Date of Birth from ID: {dob_from_id.strftime('%Y-%m-%d')}")
        age_from_id = calculate_age(dob_from_id)
        st.write(f"Age from ID: {age_from_id} years")
        
        if user_dob:
            user_age = calculate_age(user_dob)
            st.write(f"User-provided age: {user_age} years")
            
            # Allow for some discrepancy in day/month
            age_diff = abs(age_from_id - user_age)
            dob_diff = abs((dob_from_id - user_dob).days)
            
            if age_diff <= 1 and dob_diff <= 365:  # Allow 1 year difference
                st.success("✅ Age verification successful")
            else:
                st.error("❌ Age verification failed")
        else:
            st.warning("No user-provided date of birth")
    else:
        st.warning("Could not extract date of birth from ID")

    st.subheader("🏠 Address Verification")
    if user_address and user_country:
        with st.spinner("Verifying address..."):
            is_valid, details = verify_address(
                user_address, 
                user_country,
                user_state,
                user_city,
                user_postal
            )
            
            if is_valid:
                st.success("✅ Address verification successful")
                st.write("Matched to:", details)
            else:
                st.error(f"❌ Address verification failed: {details}")
    else:
        st.warning("Please provide address information for verification")

    st.subheader("🧬 Liveness Check")
    sharpness = cv2.Laplacian(cv2.cvtColor(selfie_img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    st.metric("Sharpness (Laplacian)", f"{sharpness:.2f}")
    if sharpness > 100:
        st.success("✅ Image is sharp — likely a live photo.")
    else:
        st.warning("⚠️ Image is blurry — may be spoofed.")
