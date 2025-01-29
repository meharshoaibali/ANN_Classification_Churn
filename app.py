import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import streamlit as st

# Load the model from disk
loaded_model = tf.keras.models.load_model('/workspaces/codespaces-blank/churn_model.h5')

# Load the scaler, encoder, labelencoder
with open('/workspaces/codespaces-blank/scaler', 'rb') as f:
    scaler = pickle.load(f)

with open('/workspaces/codespaces-blank/onehotencoder_geography', 'rb') as f:
    onehotencoder_geography = pickle.load(f)

with open('/workspaces/codespaces-blank/label_gender_encoder', 'rb') as f:
    label_gender_encoder = pickle.load(f)

# Streamlit app
st.title('Prediction App')

# Input fields
geography = st.selectbox('Geography', onehotencoder_geography.categories_[0])
gender = st.selectbox('Gender', label_gender_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
num_of_products = st.slider('Number of Products', 1, 4)
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data (without geography)
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_gender_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# One-hot encoding geography
geography_encoded = onehotencoder_geography.transform([[geography]])

# Getting feature names for the columns
geography_column_names = onehotencoder_geography.get_feature_names_out(['Geography'])

# Convert to a dense array and create DataFrame with the correct column names
geography_encoded_df = pd.DataFrame(geography_encoded.toarray(), columns=geography_column_names)

# Combine input_data with geography_encoded_df
input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df.reset_index(drop=True)], axis=1)


# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = loaded_model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

# Show the result
if prediction_probability > 0.5:
    st.write('Customer will leave the bank')
else:
    st.write('Customer will stay in the bank')

#run the app
