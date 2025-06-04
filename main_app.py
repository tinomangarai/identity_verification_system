# Enhanced Date of Birth Extraction
def extract_dob(text_list):
    # More comprehensive date patterns
    date_patterns = [
        r'\b(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/(19|20)\d{2}\b',  # MM/DD/YYYY
        r'\b(0?[1-9]|[12][0-9]|3[01])-(0?[1-9]|1[0-2])-(19|20)\d{2}\b',  # DD-MM-YYYY
        r'\b(0?[1-9]|[12][0-9]|3[01])\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s(19|20)\d{2}\b',  # DD Mon YYYY
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s(0?[1-9]|[12][0-9]|3[01]),\s(19|20)\d{2}\b',  # Mon DD, YYYY
        r'\b(19|20)\d{2}-(0?[1-9]|1[0-2])-(0?[1-9]|[12][0-9]|3[01])\b',  # YYYY-MM-DD
        r'\bDOB[:]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # DOB: DD/MM/YYYY
        r'\bBirth[:]?\s*(\d{1,2}\s\w+\s\d{4})\b'  # Birth: DD Mon YYYY
    ]
    
    for text in text_list:
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # Handle different date formats
                    date_str = ' '.join(matches[0]).strip()
                    
                    # Try different parsers
                    for fmt in ('%m/%d/%Y', '%d-%m-%Y', '%d %b %Y', '%b %d, %Y', 
                               '%Y-%m-%d', '%d/%m/%Y', '%m-%d-%Y'):
                        try:
                            parsed_date = datetime.strptime(date_str, fmt).date()
                            return parsed_date
                        except:
                            continue
                except Exception as e:
                    continue
    return None

# In your main form, update the address collection:
st.subheader("📝 Personal Information Verification")

col1, col2 = st.columns(2)
with col1:
    user_dob = st.date_input("Your Date of Birth", min_value=date(1900, 1, 1), 
                            help="Must match the date on your ID document")
    
with col2:
    min_age = 18
    if user_dob:
        user_age = calculate_age(user_dob)
        st.write(f"Age: {user_age} years")
        if user_age < min_age:
            st.error(f"❌ Must be at least {min_age} years old")

# Address form with better organization
with st.expander("🏠 Address Verification", expanded=True):
    st.write("Please provide your current address as it appears on your ID")
    
    user_address = st.text_input("Street Address", key="street_addr")
    user_city = st.text_input("City", key="city")
    
    col1, col2 = st.columns(2)
    with col1:
        user_state = st.text_input("State/Province", key="state")
    with col2:
        user_postal = st.text_input("Postal/Zip Code", key="postal")
    
    user_country = st.selectbox("Country", ["United States", "Canada", "United Kingdom", 
                                         "Australia", "India", "Other"])
    
    if st.button("Verify Address", key="verify_addr"):
        if not all([user_address, user_city, user_country]):
            st.warning("Please fill in all required address fields")
        else:
            with st.spinner("Validating address..."):
                is_valid, details = verify_address(
                    user_address,
                    user_country,
                    user_state if user_state else None,
                    user_city,
                    user_postal if user_postal else None
                )
                
                if is_valid:
                    st.success("✅ Address verified")
                    st.write("**Matched to:**", details)
                    # Store verification result
                    st.session_state.address_verified = True
                    st.session_state.verified_address = details
                else:
                    st.error(f"❌ Address verification failed: {details}")
                    st.session_state.address_verified = False

# Later in your verification logic:
if id_file and selfie_file:
    # ... [previous verification code] ...
    
    st.subheader("📅 Age Verification")
    dob_from_id = extract_dob(extracted_text)
    
    if dob_from_id:
        st.write(f"**Date of Birth from ID:** {dob_from_id.strftime('%Y-%m-%d')}")
        age_from_id = calculate_age(dob_from_id)
        st.write(f"**Calculated Age:** {age_from_id} years")
        
        if user_dob:
            age_diff = abs(age_from_id - calculate_age(user_dob))
            
            if age_diff == 0:
                st.success("✅ Exact age match")
            elif age_diff <= 1:
                st.success(f"✅ Approximate age match (difference: {age_diff} year)")
            else:
                st.error(f"❌ Age mismatch (difference: {age_diff} years)")
        else:
            st.warning("No user-provided date of birth for comparison")
    else:
        st.warning("Could not extract date of birth from ID")
        st.write("**Debug Info:** Common reasons:")
        st.write("- Date format not recognized")
        st.write("- Low image quality")
        st.write("- ID type not supported")
        st.write("**Extracted Text:**", ', '.join(extracted_text[:5]) + "...")
