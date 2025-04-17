import streamlit as st
from inference import BookingInference
import gdown
import os

# Ganti dengan file ID Anda
FILE_ID = "1Rmd6MC5oVjTCLab5yYUix-sBKEUVACDe" 
MODEL_PATH = "hotel_booking_cancellation_model.pkl"

if not os.path.exists(MODEL_PATH):
    try:
        gdown.download(
            f"https://drive.google.com/file/d/1Rmd6MC5oVjTCLab5yYUix-sBKEUVACDe/view?usp=sharing={FILE_ID}", 
            MODEL_PATH, 
            quiet=False
        )
    except Exception as e:
        st.error(f"Failed to download model: {str(e)}")
        st.stop()

inferencer = BookingInference(MODEL_PATH)

# Load model
inferencer = BookingInference('hotel_booking_cancellation_model.pkl')

# UI
st.title("Prediksi Pembatalan Booking Hotel")

# Input form
with st.form("input_form"):
    no_of_adults = st.number_input("Jumlah Dewasa", min_value=1, value=2)
    no_of_children = st.number_input("Jumlah Anak", min_value=0, value=0)
    lead_time = st.number_input("Lead Time (hari)", min_value=0, value=30)
    # ... (tambahkan input untuk semua feature)

    if st.form_submit_button("Prediksi"):
        input_data = {
            'no_of_adults': no_of_adults,
            'no_of_children': no_of_children,
            'lead_time': lead_time,
            # ... (lanjutkan untuk semua feature)
        }
        
        result = inferencer.predict(input_data)
        
        if result['status'] == 'success':
            st.success(f"Hasil: {result['prediction']}")
            st.metric("Probabilitas Pembatalan", f"{result['probability']:.2%}")
        else:
            st.error(f"Error: {result['status']}")
