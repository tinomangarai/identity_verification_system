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

# Initialize user information variables with Zimbabwe as default
if 'user_dob' not in st.session_state:
    st.session_state.user_dob = None
if 'user_country' not in st.session_state:
    st.session_state.user_country = "Zimbabwe"  # Default set to Zimbabwe
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

st.title("🔐 Zimbabwe Identity Verification System")

# ========== Helper Functions ==========
# [Previous helper functions remain exactly the same...]

# ========== User Interface ==========
with st.expander("📝 Personal Information", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.user_dob = st.date_input("Date of Birth", min_value=date(1900, 1, 1))
        if st.session_state.user_dob:
            age = calculate_age(st.session_state.user_dob)
            st.write(f"Age: {age} years")
            if age < 16:  # Adjusted for Zimbabwe context
                st.error("Must be 16+ for verification")
    
    with col2:
        # Zimbabwe first, then other African countries, then others
        countries = [
            "Zimbabwe",
            "South Africa",
            "Botswana",
            "Zambia",
            "Mozambique",
            "Malawi",
            "Tanzania",
            "Namibia",
            "Other"
        ]
        st.session_state.user_country = st.selectbox("Country", countries, index=0)
    
    # Zimbabwe-specific address fields
    st.session_state.user_address = st.text_input("Street Address (e.g., 123 Samora Machel Ave)", value=st.session_state.user_address)
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.user_city = st.text_input("City/Town (e.g., Harare)", value=st.session_state.user_city)
    with col2:
        st.session_state.user_state = st.text_input("Province (e.g., Mashonaland West)", value=st.session_state.user_state)
    
    st.session_state.user_postal = st.text_input("Postal Code", value=st.session_state.user_postal)
    
    if st.button("Verify Address"):
        if not all([st.session_state.user_address, st.session_state.user_city]):
            st.warning("Please fill in required fields (Street and City)")
        else:
            with st.spinner("Validating Zimbabwean address..."):
                # Special handling for Zimbabwe addresses
                if st.session_state.user_country == "Zimbabwe":
                    # Basic validation for Zimbabwe
                    valid = True
                    details = f"{st.session_state.user_address}, {st.session_state.user_city}"
                    if st.session_state.user_state:
                        details += f", {st.session_state.user_state}"
                    details += ", Zimbabwe"
                    
                    # Simple validation rules for Zimbabwe
                    if not st.session_state.user_city.lower() in ['harare', 'bulawayo', 'mutare', 'gweru', 'masvingo', 'chitungwiza', 'marondera', 'kwekwe', 'kadoma', 'chegutu', 'zvishavane', 'chinhoyi', 'bindura', 'beitbridge', 'victoria falls']:
                        valid = False
                        details = "City not recognized as major Zimbabwean city"
                    
                    if valid:
                        st.success("✅ Zimbabwean address verified")
                        st.write("Address:", details)
                        st.session_state.address_verified = True
                        st.session_state.verified_address = details
                    else:
                        st.error(f"❌ {details}")
                        st.session_state.address_verified = False
                else:
                    # Original verification for other countries
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

# [Rest of the code remains exactly the same...]
