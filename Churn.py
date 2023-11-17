import streamlit as st
import numpy as np
import pickle
from tensorflow import keras

# Load the saved scaler model
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Specify the path to your saved model
best_model_path = "best_model.h5"

# Load the saved model
best_model = keras.models.load_model(best_model_path)

# Define the input form
st.title('Customer Churn Prediction')
st.write('Please enter the following information to predict whether a customer is likely to churn:')

# Feature explanations
st.sidebar.title('Feature Explanations:')
st.sidebar.markdown('- **Total Charges:** Total charges incurred by the customer.')
st.sidebar.markdown('- **Monthly Charges:** Monthly charges paid by the customer.')
st.sidebar.markdown('- **Tenure:** Number of months the customer has stayed with the company.')
st.sidebar.markdown('- **Contract Type:** Type of contract the customer has (Month-to-month, One year, Two year).')
st.sidebar.markdown('- **Payment Method:** Customer\'s preferred payment method (Electronic check, Bank transfer, Mailed check).')
st.sidebar.markdown('- **Tech Support:** Whether the customer has tech support (Yes, No).')
st.sidebar.markdown('- **Online Security:** Whether the customer has online security (Yes, No).')
st.sidebar.markdown('- **Internet Service:** Type of internet service (DSL, Fiber optic, No).')
st.sidebar.markdown('- **Online Backup:** Whether the customer has online backup (Yes, No).')
st.sidebar.markdown('- **Gender:** Gender of the customer (Male, Female).')

# Collect user inputs
total_charges = st.number_input('Total charges:')
monthly_charges = st.number_input('Monthly charges:')
tenure = st.number_input('Tenure:')
contract = st.selectbox('Contract type:', ['Month-to-month', 'One year', 'Two years'])
payment_method = st.selectbox('Payment method:', ['Electronic check', 'Bank transfer', 'Mailed check'])
tech_support = st.selectbox('Tech support:', ['Yes', 'No'])
online_security = st.selectbox('Online security:', ['Yes', 'No'])
internet_service = st.selectbox('Internet service:', ['DSL', 'Fiber optic', 'No'])
online_backup = st.selectbox('Online backup:', ['Yes', 'No'])
gender = st.selectbox('Gender:', ['Male', 'Female'])

# Create a button for prediction
if st.button('Predict'):
    # Prepare the input data
    input_data = np.array([total_charges, monthly_charges, tenure])

    # Convert categorical features to one-hot encoding
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two years': 2}
    payment_method_mapping = {'Electronic check': 0, 'Bank transfer': 1, 'Mailed check': 2}
    tech_support_mapping = {'Yes': 1, 'No': 0}
    online_security_mapping = {'Yes': 1, 'No': 0}
    internet_service_mapping = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    online_backup_mapping = {'Yes': 1, 'No': 0}
    gender_mapping = {'Male': 0, 'Female': 1}

    contract_encoded = contract_mapping.get(contract, -1)
    payment_method_encoded = payment_method_mapping.get(payment_method, -1)
    tech_support_encoded = tech_support_mapping.get(tech_support, -1)
    online_security_encoded = online_security_mapping.get(online_security, -1)
    internet_service_encoded = internet_service_mapping.get(internet_service, -1)
    online_backup_encoded = online_backup_mapping.get(online_backup, -1)
    gender_encoded = gender_mapping.get(gender, -1)

    if (
        contract_encoded == -1
        or payment_method_encoded == -1
        or tech_support_encoded == -1
        or online_security_encoded == -1
        or internet_service_encoded == -1
        or online_backup_encoded == -1
        or gender_encoded == -1
    ):
        st.write("Invalid input for categorical features.")
    else:
        input_data = np.concatenate(
            [input_data, [contract_encoded, payment_method_encoded, tech_support_encoded, online_security_encoded,
                          internet_service_encoded, online_backup_encoded, gender_encoded]])

        # Scale the input data using the loaded scaler
        scaled_data = scaler.transform(input_data.reshape(1, -1))

        # Make the prediction using the loaded model
        prediction = best_model.predict(scaled_data)

        # Display the prediction result
        if prediction > 0.5:  # Assuming the model returns probabilities
            result = 'This customer is likely to churn.'
        else:
            result = 'This customer is not likely to churn.'

        st.write(result)
