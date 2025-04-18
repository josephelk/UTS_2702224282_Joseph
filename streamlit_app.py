import streamlit as st
import pandas as pd
import joblib

# Load model
# Model IPYNB
# model = joblib.load("hotel_booking_cancellation_model.pkl")
# Model OOP
model = joblib.load("rf_booking_model.pkl")


def user_input_form():
    with st.form("booking_form"):
        # Input columns that match the dataset
        no_of_adults = st.number_input("Number of Adults", min_value=0, value=1)
        no_of_children = st.number_input("Number of Children", min_value=0, value=1)
        no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0,value=1)
        no_of_week_nights = st.number_input("Number of Week Nights", min_value=0,value=2)
        type_of_meal_plan = st.selectbox("Type of Meal Plan", ['Not Selected', 'Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3'])
        required_car_parking_space = st.selectbox("Required Car Parking Space", [0, 1])
        room_type_reserved = st.selectbox("Room Type Reserved", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
        lead_time = st.number_input("Lead Time", min_value=0, value=1)
        market_segment_type = st.selectbox("Market Segment Type", ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary'])
        repeated_guest = st.selectbox("Repeated Guest", [0, 1])
        no_of_previous_cancellations = st.number_input("Number of Previous Cancellations", min_value=0, value=0)
        no_of_previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, value=0)
        avg_price_per_room = st.number_input("Average Price Per Room", min_value=0.0, value=100.0)
        no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, value=0)
        total_guests = no_of_adults + no_of_children
        total_nights = no_of_weekend_nights + no_of_week_nights
        st.text_input("Total Guests", value=str(total_guests), disabled=True)
        st.text_input("Total Nights", value=str(total_nights), disabled=True)
        booking_season = st.selectbox("Booking Season", ['Fall', 'Winter', 'Spring', 'Summer'])

        submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Return input data as dictionary
        return {
            "no_of_adults": no_of_adults,
            "no_of_children": no_of_children,
            "no_of_weekend_nights": no_of_weekend_nights,
            "no_of_week_nights": no_of_week_nights,
            "type_of_meal_plan": type_of_meal_plan,
            "required_car_parking_space": required_car_parking_space,
            "room_type_reserved": room_type_reserved,
            "lead_time": lead_time,
            "market_segment_type": market_segment_type,
            "repeated_guest": repeated_guest,
            "no_of_previous_cancellations": no_of_previous_cancellations,
            "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
            "avg_price_per_room": avg_price_per_room,
            "no_of_special_requests": no_of_special_requests,
            "total_guests": total_guests,
            "booking_season": booking_season,
            "total_nights": total_nights
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
