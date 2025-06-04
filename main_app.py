import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, date
import requests
import re
from functools import lru_cache

# Initialize session state
if 'address_verified' not in st.session_state:
    st.session_state.address_verified = False
if 'face_verified' not in st.session_state:
    st.session_state.face_verified = False

# Load OCR (cached)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

st.title("🔐 Enhanced Identity Verification System")

# ========== Helper Functions ==========
def read_image(file):
    """Load image from file uploader and resize it"""
    image = np.array(Image.open(file).convert('RGB'))
    # Resize image to a maximum width or height of 800 pixels
    max_size = 800
    min_size = 100  # Minimum size to prevent very small images
    height, width, _ = image.shape
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    # Ensure image is not smaller than the minimum size
    if height < min_size or width < min_size:
        image = cv2.resize(image, (min_size, min_size), interpolation=cv2.INTER_AREA)
    
    return image

def detect_face(image_np):
    """Detect face using OpenCV Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return image_np[y:y+h, x:x+w], (x, y, w, h)

def compare_faces(face1, face2):
    """Compare faces using simple pixel comparison"""
    face1_resized = cv2.resize(face1, (100, 100))
    face2_resized = cv2.resize(face2, (100, 100))
    difference = cv2.absdiff(face1_resized, face2_resized)
    similarity = np.sum(difference) / (100 * 100 * 3)  # Average pixel difference
    threshold = 50  # Set a threshold for matching
    return similarity < threshold

def extract_dob(text_list):
    """Extract date of birth from OCR text"""
    date_patterns = [
        (r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](19|20)\d{2}\b', ['%m/%d/%Y', '%m-%d-%Y']),
        (r'\b(0?[1-9]|[12][0-9]|3[01])[/-](0?[1-9]|1[0-2])[/-](19|20)\d{2}\b', ['%d/%m/%Y', '%d-%m-%Y']),
        (r'\b(19|20)\d{2}[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])\b', ['%Y/%m/%d', '%Y-%m-%d']),
        (r'\b(DOB|Birth|Date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', ['%m/%d/%Y', '%d/%m/%Y'])
    ]
    
    for text in text_list:
        for pattern, formats in date_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                date_str = matches.group()
                for fmt in formats:
                    try:
                        return datetime.strptime(date_str, fmt).date()
                    except ValueError:
                        continue
    return None

def calculate_age(dob):
    """Calculate age from date of birth"""
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

@lru_cache(maxsize=100)
def verify_address(street, country, city, state=None, postal=None):
    """Validate address using Google Geocoding API"""
    try:
        if not all([street, city, country]):
            return False, "Missing required fields (Street, City, Country)"
        
        query = f"{street}, {city}, {state}, {postal}, {country}"
        params = {
            'address': query,
            'key': 'AIzaSyBcTIILXOzrXyjYh-BjAWoH0dh_NReVVcw'  # Your API key
        }
        
        response = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'OK':
            return True, data['results'][0]['formatted_address']
        return False, "Address not found"
    except Exception as e:
        return False, f"Error: {str(e)}"

# ========== User Interface ==========
with st.expander("📝 Personal Information", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        user_dob = st.date_input("Date of Birth", min_value=date(1900, 1, 1))
        if user_dob:
            age = calculate_age(user_dob)
            st.write(f"Age: {age} years")
            if age < 18:
                st.error("Must be 18+ for verification")
    
    with col2:
        countries = ["Zimbabwe", "United States", "Canada", "United Kingdom", "Australia", "Germany", "France", "Other"]
        user_country = st.selectbox("Country", countries, index=0)
    
    user_address = st.text_input("Street Address (e.g., 123 Main St)")
    col1, col2 = st.columns(2)
    with col1:
        user_city = st.text_input("City (e.g., Harare)")
    with col2:
        user_state = st.text_input("State/Province (optional)")
    user_postal = st.text_input("Postal/Zip Code (optional)")
    
    if st.button("Verify Address"):
        with st.spinner("Validating address..."):
            is_valid, details = verify_address(
                user_address,
                user_country,
                user_city,
                user_state if user_state.strip() else None,
                user_postal if user_postal.strip() else None
            )
            
            if is_valid:
                st.success("✅ Address verified")
                st.write("Matched to:", details)
                st.session_state.address_verified = True
            else:
                st.error(f"❌ {details}")
                st.session_state.address_verified = False

st.divider()

# ========== Document Verification ==========
st.subheader("📄 Document Upload")
col1, col2 = st.columns(2)
with col1:
    id_file = st.file_uploader("ID Document", type=["jpg", "jpeg", "png"])
with col2:
    selfie_file = st.file_uploader("Selfie Photo", type=["jpg", "jpeg", "png"])

if id_file and selfie_file:
    id_img = read_image(id_file)
    selfie_img = read_image(selfie_file)

    # OCR Processing
    st.subheader("🔍 ID Document Analysis")
    with st.spinner("Extracting text..."):
        ocr_results = reader.readtext(id_img)
        extracted_text = [result[1] for result in ocr_results]
        
        if extracted_text:
            st.success(f"Extracted {len(extracted_text)} text elements")
            with st.expander("View extracted text"):
                for text in extracted_text[:10]:
                    st.write(f"- {text}")
        else:
            st.warning("No text found in ID document")

    # Face Verification
    st.subheader("🧑 Face Verification")
    face_id, face_id_coords = detect_face(id_img)
    face_selfie, face_selfie_coords = detect_face(selfie_img)

    if face_id is None or face_selfie is None:
        st.error("Could not detect faces in one or both images")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(face_id, caption="ID Photo Face", width=300)  # Fixed width for visibility
        with col2:
            st.image(face_selfie, caption="Selfie Face", width=300)  # Fixed width for visibility
        
        with st.spinner("Comparing faces..."):
            match = compare_faces(face_id, face_selfie)
        
        if match is not None:
            if match:
                st.success("✅ Faces match!")
                st.session_state.face_verified = True
            else:
                st.error("❌ Faces do not match.")
                st.session_state.face_verified = False

    # Age Verification
    st.subheader("📅 Age Verification")
    dob_from_id = extract_dob(extracted_text)
    
    if dob_from_id:
        st.write(f"Detected DOB on ID: {dob_from_id.strftime('%Y-%m-%d')}")
        age_from_id = calculate_age(dob_from_id)
        st.write(f"Age from ID: {age_from_id} years")
        
        if user_dob:
            age_diff = abs(age_from_id - calculate_age(user_dob))
            if age_diff == 0:
                st.success("✅ Exact age match")
            elif age_diff <= 1:
                st.success(f"✅ Approximate match ({age_diff} year difference)")
            else:
                st.error(f"❌ Age mismatch ({age_diff} years difference)")
        else:
            st.warning("No user-provided DOB for comparison")
    else:
        st.error("Could not extract date of birth from ID")

    # Liveness Check
    st.subheader("🧬 Liveness Detection")
    sharpness = cv2.Laplacian(cv2.cvtColor(selfie_img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    st.metric("Image Sharpness", f"{sharpness:.2f}")
    if sharpness > 100:
        st.success("✅ Good sharpness - likely live photo")
    elif sharpness > 50:
        st.warning("⚠️ Moderate sharpness - possible screenshot")
    else:
        st.error("❌ Low sharpness - potential spoof")

    # Final Verification Summary
    st.subheader("🎯 Verification Summary")
    
    verification_passed = True
    if not st.session_state.face_verified:
        verification_passed = False
        st.error("❌ Face verification failed")
    
    if dob_from_id and user_dob and abs(calculate_age(dob_from_id) - calculate_age(user_dob)) > 1:
        verification_passed = False
        st.error("❌ Age verification failed")
    
    if not st.session_state.address_verified:
        verification_passed = False
        st.error("❌ Address not verified")
    
    if sharpness < 50:
        verification_passed = False
        st.error("❌ Liveness check failed")
    
    if verification_passed:
        st.success("✅ All verifications passed!")
        st.balloons()
    else:
        st.error("❌ Verification incomplete - please check failed items")
