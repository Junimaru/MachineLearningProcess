import streamlit as st
import requests
import joblib
from PIL import Image

# Load and set images in the first place
header_images = Image.open("assets/waterpot.jpg")
st.image(header_images)

# Add some information about the service
st.title("Water Potability Prediction")
st.subheader("Please enter variabel below then click Predict button to get an amazing result!!!")

# Create form of input
with st.form(key = "water_component_data_form"): 
    # Create box for number input
    ph = st.number_input("1.Enter pH Value (float):",
        min_value = 0.0,
        max_value = 15.0,
        help = "Value range from 0 to 15"
    )

    hardness = st.number_input("2.Enter Hardness Value (float):",
        min_value = 45.0,
        max_value = 350.0,
        help = "Value range from 45 to 350"
    )
    
    solids = st.number_input("3.Enter Solids Value (float):",
        min_value = 300.0,
        max_value = 70000.0,
        help = "Value range from 300 to 70.000"
    )

    chloramines = st.number_input("4.Enter Chloramines Value (float):",
        min_value = 0.0,
        max_value = 15.0,
        help = "Value range from 0 to 15"
    )

    sulfate= st.number_input("5.Enter Sulfate Value (float):",
        min_value = 100.0,
        max_value = 500.0,
        help = "Value range from 100 to 500"
    )

    conductivity = st.number_input("6.Enter Conductivity Value (float):",
        min_value = 100.0,
        max_value = 1000.0,
        help = "Value range from 100 to 1.000"
    )

    organic_carbon = st.number_input("7.Enter Organic Carbon Value (float):",
        min_value = 0.0,
        max_value = 50.0,
        help = "Value range from 0 to 50"
    )

    trihalomethanes = st.number_input("8.Enter Raw Trihalomethanes Value (float):",
        min_value = 0.0,
        max_value = 150.0,
        help = "Value range from 0 to 150"
    )

    turbidity = st.number_input("9.Enter Raw Turbidity Value (float):",
        min_value = 0.0,
        max_value = 10.0,
        help = "Value range from 0 to 10"
    )
    
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "ph": ph,
            "Hardness": hardness,
            "Solids": solids,
            "Chloramines": chloramines,
            "Sulfate": sulfate,
            "Conductivity": conductivity,
            "Organic_carbon": organic_carbon,
            "Trihalomethanes": trihalomethanes,
            "Turbidity": turbidity
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://localhost:8080/predict/", json = raw_data).json() #http://api_backend:8080/
            
        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "Not Potable.":
                st.warning("Potable.")
            else:
                st.success("Not Potable.")
