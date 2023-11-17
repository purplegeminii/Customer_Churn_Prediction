import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import tensorflow
from tensorflow import keras
from keras.models import load_model, Model
from sklearn.preprocessing import LabelEncoder


# Load the pickled encoders_dict
encoders_dict = pickle.load(open('encoders_dict.pkl', 'rb'))
print(encoders_dict)

# load the pickled scaler
scaler = pickle.load(open('x_scaler.pkl', 'rb'))

# load the pickled model
model = load_model('MLP_grid_model.h5')

st.title('CUSTOMER CHURN PREDICTION')

Dependents = st.radio('Dependents', ('No', 'Yes'))
tenure = int(st.number_input('tenure'))
OnlineSecurity = st.radio('OnlineSecurity', ('No', 'Yes', 'No internet service'))
OnlineBackup = st.radio('OnlineBackup', ('No', 'Yes', 'No internet service'))
DeviceProtection = st.radio('DeviceProtection', ('No', 'Yes', 'No internet service'))
TechSupport = st.radio('TechSupport', ('No', 'Yes', 'No internet service'))
Contract = st.radio('Contract', ('Month-to-month', 'Two year', 'One year'))
PaperlessBilling = st.radio('PaperlessBilling', ('Yes', 'No'))
MonthlyCharges = float(st.number_input('MonthlyCharges'))
TotalCharges = float(st.number_input('TotalCharges'))

user_inputs_list = [Dependents, tenure, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, Contract, PaperlessBilling, MonthlyCharges, TotalCharges]
user_inputs_list = np.array(user_inputs_list)

user_inputs = pd.DataFrame(user_inputs_list.reshape(1, -1), columns=['Dependents', 'tenure', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'])

# Encode the categorical variables
for col in user_inputs.columns:
    if col in encoders_dict.keys():
        user_inputs[col] = encoders_dict[col].transform(user_inputs[col])

# Scale the numerical variables
user_inputs = scaler.transform(user_inputs)

# Make predictions
prediction = model.predict(user_inputs)

if st.button('SUBMIT'):
    if prediction[0][0] >= 0.5:
        st.write('Customer will churn')
    else:
        st.write('Customer will not churn')
