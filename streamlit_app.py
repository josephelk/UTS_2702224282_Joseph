import streamlit as st
import joblib
import pandas as pd



import os

model_path = 'hotel_booking_cancellation_model.pkl'  # Path ke model yang ingin dimuat

if os.path.exists(model_path):
    print("Model file found!")
else:
    print(f"Model file {model_path} not found!")

# Load model
model = joblib.load("rf_booking_model.pkl")  # Model yang sudah dilatih

def user_input_form():
    with st.form("booking_form"):
        # Input fitur yang diperlukan
        no_of_adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=2)
        no_of_children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
        no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, max_value=7, value=1)
        no_of_week_nights = st.number_input("Number of Week Nights", min_value=1, max_value=7, value=3)
        type_of_meal_plan = st.selectbox("Meal Plan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3'])
        required_car_parking_space = st.selectbox("Car Parking Space Required", [0, 1])
        room_type_reserved = st.selectbox("Room Type Reserved", ['Room Type 1', 'Room Type 2', 'Room Type 3'])
        lead_time = st.number_input("Lead Time", min_value=1, max_value=365, value=10)
        repeated_guest = st.selectbox("Is Repeated Guest?", [0, 1])
        no_of_previous_cancellations = st.number_input("Number of Previous Cancellations", min_value=0, max_value=10, value=0)
        no_of_previous_bookings_not_canceled = st.number_input("Number of Previous Bookings Not Canceled", min_value=0, max_value=10, value=1)
        avg_price_per_room = st.number_input("Average Price Per Room (in Euros)", min_value=0.0, value=100.0)
        no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, max_value=5, value=1)
        booking_season = st.selectbox("Booking Season", ['Winter', 'Spring', 'Summer', 'Fall'])

        submitted = st.form_submit_button("Predict")

    if submitted:
        return {
            "no_of_adults": no_of_adults,
            "no_of_children": no_of_children,
            "no_of_weekend_nights": no_of_weekend_nights,
            "no_of_week_nights": no_of_week_nights,
            "type_of_meal_plan": type_of_meal_plan,
            "required_car_parking_space": required_car_parking_space,
            "room_type_reserved": room_type_reserved,
            "lead_time": lead_time,
            "repeated_guest": repeated_guest,
            "no_of_previous_cancellations": no_of_previous_cancellations,
            "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
            "avg_price_per_room": avg_price_per_room,
            "no_of_special_requests": no_of_special_requests,
            "booking_season": booking_season
        }
    else:
        return None

def predict_booking_status(input_dict):
    df = pd.DataFrame([input_dict])

    # Jika perlu, konversi fitur kategorikal (seperti 'type_of_meal_plan', 'room_type_reserved', 'booking_season')
    # menjadi variabel dummy (One Hot Encoding) jika model menginginkannya
    df = pd.get_dummies(df, columns=["type_of_meal_plan", "room_type_reserved", "booking_season"], drop_first=True)

    # Prediksi status booking
    pred = model.predict(df)[0]
    return pred

def main():
    st.title("Hotel Booking Cancellation Prediction App")
    st.write("Input your booking data below to predict the cancellation status.")

    user_data = user_input_form()

    if user_data:
        result = predict_booking_status(user_data)
        if result == 1:
            st.success("Your booking is Canceled")
        else:
            st.success("Your booking is Not Canceled")

if __name__ == "__main__":
    main()


