import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("savedPickle/model.pkl")

def user_input_form():
    with st.form("booking_form"):
        # Input kolom yang dibutuhkan oleh model
        total_guests = st.number_input("Total Guests", min_value=1, value=1)
        market_segment_type = st.selectbox("Market Segment", ['Online', 'Corporate', 'Direct', 'Offline', 'Other'])
        total_nights = st.number_input("Total Nights", min_value=1, value=1)
        lead_time = st.number_input("Lead Time", min_value=1, value=10)
        no_of_previous_cancellations = st.number_input("Number of Previous Cancellations", min_value=0, value=0)
        room_type = st.selectbox("Room Type", ['Single', 'Double', 'Suite', 'Penthouse'])
        booking_changes = st.number_input("Booking Changes", min_value=0, value=0)
        deposit_type = st.selectbox("Deposit Type", ['No Deposit', 'Refundable', 'Non Refundable'])
        agent = st.number_input("Agent", min_value=1, value=1)
        customer_type = st.selectbox("Customer Type", ['Contract', 'Transient', 'Group'])
        previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, value=0)
        booking_date = st.date_input("Booking Date")
        cancellation_date = st.date_input("Cancellation Date")
        lead_time = (pd.to_datetime(cancellation_date) - pd.to_datetime(booking_date)).days

        submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Return input data as dictionary
        return {
            "total_guests": total_guests,
            "market_segment_type": market_segment_type,
            "total_nights": total_nights,
            "lead_time": lead_time,
            "no_of_previous_cancellations": no_of_previous_cancellations,
            "room_type": room_type,
            "booking_changes": booking_changes,
            "deposit_type": deposit_type,
            "agent": agent,
            "customer_type": customer_type,
            "previous_bookings_not_canceled": previous_bookings_not_canceled,
            "booking_date": booking_date,
            "cancellation_date": cancellation_date
        }
    else:
        return None

def predict_booking_status(input_dict):
    df = pd.DataFrame([input_dict])

    # Asumsikan model membutuhkan beberapa kolom untuk encoding atau preprocessing
    # Jika model memiliki pengolahan seperti encoding atau normalisasi, lakukan di sini
    # df = preprocessing(df)

    # Model prediction
    pred = model.predict(df)[0]
    return pred

def main():
    st.title("Hotel Booking Cancellation Prediction")
    st.write("Input your data below to predict if your booking will be canceled or not.")

    user_data = user_input_form()

    if user_data:
        result = predict_booking_status(user_data)
        if result == 1:
            st.success("Your booking is likely to be Canceled.")
        else: 
            st.success("Your booking is likely to be Not Canceled.")

if __name__ == "__main__":
    main()
