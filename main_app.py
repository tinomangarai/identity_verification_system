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

# Initialize session state
if 'address_verified' not in st.session_state:
    st.session_state.address_verified = False
    st.session_state.verified_address = None

# Initialize user information variables
if 'user_dob' not in st.session_state:
    st.session_state.user_dob = None
if 'user_country' not in st.session_state:
    st.session_state.user_country = "United States"  # Default value
if 'user_address' not in st.session_state:
    st.session_state.user_address = ""
if 'user_city' not in st.session_state:
    st.session_state.user_city = ""
if 'user_state' not in st.session_state:
    st.session_state.user_state = ""
if 'user_postal' not in st.session_state:
    st.session_state.user_postal = ""

# Load OCR
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

st.title("🔐 Enhanced Identity Verification System")

# ========== Helper Functions ==========
def read_image(file):
    """Load image from file uploader"""
    return np.array(Image.open(file).convert('RGB'))

def detect_face(image_np):
    """Detect face using Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return image_np[y:y+h, x:x+w]

def compare_faces(face1, face2):
    """Compare faces using cosine similarity"""
    try:
        face1_resized = cv2.resize(face1, (100, 100)).flatten().reshape(1, -1)
        face2_resized = cv2.resize(face2, (100, 100)).flatten().reshape(1, -1)
        similarity = cosine_similarity(face1_resized, face2_resized)[0][0]
        return similarity
    except Exception as e:
        st.error(f"Face comparison error: {str(e)}")
        return None

def extract_dob(text_list):
    """Improved date of birth extraction with multiple patterns"""
    date_patterns = [
        (r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](19|20)\d{2}\b', ['%m/%d/%Y', '%m-%d-%Y']),
        (r'\b(0?[1-9]|[12][0-9]|3[01])[/-](0?[1-9]|1[0-2])[/-](19|20)\d{2}\b', ['%d/%m/%Y', '%d-%m-%Y']),
        (r'\b(19|20)\d{2}[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])\b', ['%Y/%m/%d', '%Y-%m-%d']),
        (r'\b(0?[1-9]|[12][0-9]|3[01])\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s(19|20)\d{2}\b', ['%d %b %Y']),
        (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s(0?[1-9]|[12][0-9]|3[01]),\s(19|20)\d{2}\b', ['%b %d, %Y']),
        (r'\b(DOB|Birth|Date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', ['%m/%d/%Y', '%d/%m/%Y']),
        (r'\b\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b', ['%d %B %Y'])
    ]

    for text in text_list:
        for pattern, formats in date_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                date_str = matches.group()
                for fmt in formats:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt).date()
                        return parsed_date
                    except ValueError:
                        continue
    return None

def calculate_age(dob):
    """Calculate age from date of birth"""
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

def verify_address(street, country, city, state=None, postal=None):
    """Validate address using Nominatim API"""
    try:
        query_parts = {
            'street': street,
            'city': city,
            'state': state,
            'postalcode': postal,
            'country': country,
            'format': 'json',
            'limit': 1
        }
        query = {k: v for k, v in query_parts.items() if v}
        
        headers = {'User-Agent': 'IDVerificationApp/1.0'}
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=query,
            headers=headers
        )
        response.raise_for_status()
        
        data = response.json()
        if data:
            return True, data[0]['display_name']
        return False, "Address not found in database"
    except Exception as e:
        return False, f"Verification error: {str(e)}"

# ========== User Interface ==========
with st.expander("📝 Personal Information", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.user_dob = st.date_input("Date of Birth", min_value=date(1900, 1, 1))
        if st.session_state.user_dob:
            age = calculate_age(st.session_state.user_dob)
            st.write(f"Age: {age} years")
            if age < 18:
                st.error("Must be 18+ for verification")
    
    with col2:
        countries = ["United States", "Canada", "United Kingdom", "Australia", "Germany", "France", "Other"]
        st.session_state.user_country = st.selectbox("Country", countries, index=0)
    
    st.session_state.user_address = st.text_input("Street Address", value=st.session_state.user_address)
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.user_city = st.text_input("City", value=st.session_state.user_city)
    with col2:
        st.session_state.user_state = st.text_input("State/Province (if applicable)", value=st.session_state.user_state)
    
    st.session_state.user_postal = st.text_input("Postal/Zip Code", value=st.session_state.user_postal)
    
    if st.button("Verify Address"):
        if not all([st.session_state.user_address, st.session_state.user_city, st.session_state.user_country]):
            st.warning("Please fill in required fields (Street, City, Country)")
        else:
            with st.spinner("Validating address..."):
                is_valid, details = verify_address(
                    st.session_state.user_address,
                    st.session_state.user_country,
                    st.session_state.user_city,
                    st.session_state.user_state if st.session_state.user_state else None,
                    st.session_state.user_postal if st.session_state.user_postal else None
                )
                if is_valid:
                    st.success("✅ Address verified")
                    st.write("Matched to:", details)
                    st.session_state.address_verified = True
                    st.session_state.verified_address = details
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
    try:
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
        face_id = detect_face(id_img)
        face_selfie = detect_face(selfie_img)

        if face_id is None or face_selfie is None:
            st.error("Could not detect faces in one or both images")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(face_id, caption="ID Photo Face", use_column_width=True)
            with col2:
                st.image(face_selfie, caption="Selfie Face", use_column_width=True)
            
            similarity = compare_faces(face_id, face_selfie)
            if similarity is not None:
                st.metric("Face Match Score", f"{similarity:.2%}")
                if similarity > 0.75:
                    st.success("✅ High similarity - likely match")
                elif similarity > 0.5:
                    st.warning("⚠️ Moderate similarity - review needed")
                else:
                    st.error("❌ Low similarity - possible mismatch")

        # Age Verification
        st.subheader("📅 Age Verification")
        dob_from_id = extract_dob(extracted_text)
        
        if dob_from_id:
            st.write(f"Detected DOB on ID: {dob_from_id.strftime('%Y-%m-%d')}")
            age_from_id = calculate_age(dob_from_id)
            st.write(f"Age from ID: {age_from_id} years")
            
            if st.session_state.user_dob:
                age_diff = abs(age_from_id - calculate_age(st.session_state.user_dob))
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
            with st.expander("Debug Info"):
                st.write("Common reasons:")
                st.write("- Date format not recognized")
                st.write("- Low image quality")
                st.write("- ID type not supported")
                st.write("Sample extracted text:", ', '.join(extracted_text[:3]) + "...")

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
        st.divider()
        st.subheader("🎯 Verification Summary")
        
        verification_passed = True
        if similarity and similarity < 0.6:
            verification_passed = False
            st.error("❌ Face verification failed")
        
        if dob_from_id and st.session_state.user_dob and abs(calculate_age(dob_from_id) - calculate_age(st.session_state.user_dob)) > 1:
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

    except Exception as e:
        st.error(f"An error occurred during verification: {str(e)}")
        st.error("Please try again or contact support")
