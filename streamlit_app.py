import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('rf_booking_model.pkl')

# Function to get booking season based on month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Streamlit UI for input
def user_input_features():
    st.title('Hotel Booking Status Prediction')
    
    # Input fields for users
    no_of_adults = st.number_input('Number of Adults', min_value=1, max_value=10, value=2)
    no_of_children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
    no_of_weekend_nights = st.number_input('Number of Weekend Nights', min_value=0, max_value=7, value=1)
    no_of_week_nights = st.number_input('Number of Week Nights', min_value=0, max_value=7, value=1)
    type_of_meal_plan = st.selectbox('Meal Plan', ['No Meal', 'Breakfast', 'Half Board', 'Full Board'])
    required_car_parking_space = st.selectbox('Required Car Parking Space', ['Yes', 'No'])
    room_type_reserved = st.selectbox('Room Type Reserved', ['Reserved Room 1', 'Reserved Room 2', 'Reserved Room 3'])
    lead_time = st.number_input('Lead Time (Days)', min_value=0, max_value=365, value=30)
    arrival_month = st.number_input('Arrival Month', min_value=1, max_value=12, value=5)
    repeated_guest = st.selectbox('Repeated Guest', ['Yes', 'No'])
    no_of_previous_cancellations = st.number_input('Previous Cancellations', min_value=0, value=0)
    no_of_previous_bookings_not_canceled = st.number_input('Previous Bookings Not Canceled', min_value=0, value=0)
    avg_price_per_room = st.number_input('Average Price per Room (Euro)', min_value=10, value=100)
    no_of_special_requests = st.number_input('Number of Special Requests', min_value=0, max_value=5, value=0)

    # Calculate additional features based on inputs
    total_guests = no_of_adults + no_of_children
    total_nights = no_of_weekend_nights + no_of_week_nights
    booking_season = get_season(arrival_month)

    # Map inputs to a dictionary
    data = {
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': type_of_meal_plan,
        'required_car_parking_space': 1 if required_car_parking_space == 'Yes' else 0,
        'room_type_reserved': room_type_reserved,
        'lead_time': lead_time,
        'arrival_month': arrival_month,
        'repeated_guest': 1 if repeated_guest == 'Yes' else 0,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests,
        'total_guests': total_guests,  # Added missing column
        'total_nights': total_nights,  # Added missing column
        'booking_season': booking_season  # Added missing column
    }
    
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame(data, index=[0])
    return input_df

# Prepare the input features for prediction
user_input = user_input_features()

# Predict the booking status (canceled or not canceled)
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)[:, 1]

# Display prediction results
st.write("### Prediction Result:")
if prediction == 1:
    st.write("The booking is likely to be **Canceled**.")
else:
    st.write("The booking is likely to be **Not Canceled**.")

st.write(f"Prediction Probability (Canceled): {prediction_proba[0]:.2f}")

# Add a button to allow user to input new data and make predictions again
st.button("Make another prediction")
