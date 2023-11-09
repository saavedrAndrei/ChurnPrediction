from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set the layout to wide to remove the small blank space in the sidebar
st.title("🎯 Telco Customer Churn Prediction")
st.write("👤 by `Andrei Saavedra`")
st.write("[GitHub Dataset URL](https://github.com/saavedrAndrei) 🚀")

# Description
st.write("The churn rate, also known as the rate of attrition or customer churn, is the rate at which customers stop doing business with an entity (e.g., Business Organization).")

# Model Information
st.write("You will predict the churn rate of customers in a telecom company using a stored model based on XGBoost, CatBoost, or LightGBM.")

# Sidebar
st.header("Instructions")
st.write("1. The model is trained on a XGBoost Classifier.")
st.write("2. To check the accuracy, click on 'Performance on Test Dataset'.")
st.write("3. To predict churn rate by manual input, scroll down and click 'Predict' button in the.")
st.write("The result will be displayed in the 'Prediction Result' section.")

# Dataset Source
st.write("Dataset Source:")
st.write("1. [Kaggle Dataset URL](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")
st.write("2. [GitHub Dataset URL](https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data)")


# Display the user input data
# Define select box options
gender_options = ['Male', 'Female']
yes_no_options = ['Yes', 'No']
internet_service_options = ['DSL', 'Fiber optic', 'No']
online_security_options = ['Yes', 'No', 'No internet service']
contract_options = ['Month-to-month', 'One year', 'Two year']
payment_method_options = ['Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check']

user_data = {}

# Create a Streamlit sidebar
st.sidebar.header("**Predict Customer Churn Rate**")

st.sidebar.header("User Input")
user_data['gender'] = st.sidebar.selectbox('Gender', gender_options)
user_data['SeniorCitizen'] = st.sidebar.selectbox('Senior Citizen', yes_no_options)
user_data['Partner'] = st.sidebar.selectbox('Partner', yes_no_options)
user_data['Dependents'] = st.sidebar.selectbox('Dependents', yes_no_options)
user_data['PhoneService'] = st.sidebar.selectbox('Phone Service', yes_no_options)
user_data['MultipleLines'] = st.sidebar.selectbox('Multiple Lines', yes_no_options)
user_data['InternetService'] = st.sidebar.selectbox('Internet Service Type', internet_service_options)
user_data['OnlineSecurity'] = st.sidebar.selectbox('Online Security', online_security_options)
user_data['OnlineBackup'] = st.sidebar.selectbox('Online Backup', online_security_options)
user_data['DeviceProtection'] = st.sidebar.selectbox('Device Protection', online_security_options)
user_data['TechSupport'] = st.sidebar.selectbox('Tech Support', online_security_options)
user_data['StreamingTV'] = st.sidebar.selectbox('Streaming TV', online_security_options)
user_data['StreamingMovies'] = st.sidebar.selectbox('Streaming Movies', online_security_options)
user_data['Contract'] = st.sidebar.selectbox('Contract', contract_options)
user_data['PaperlessBilling'] = st.sidebar.selectbox('Paperless Billing', yes_no_options)
user_data['PaymentMethod'] = st.sidebar.selectbox('Payment Method', payment_method_options)
user_data['tenure'] = st.sidebar.slider('Tenure', min_value=0, max_value=72)
user_data['MonthlyCharges'] = st.sidebar.slider('Monthly Charges', min_value=0.0, max_value=720.0)
user_data['TotalCharges'] = st.sidebar.slider('Total Charges', min_value=0.0, max_value=8640.0)

st.write("## Churn Data Overview")
st.write("Data Dimension: 7043 rows and 24 columns.")
original_df = pd.read_csv('telco-customer-data.csv')
st.write(original_df)
# Create a hyperlink to download the dataset


# Display the user input data
st.write("## User Input Data")
user_df = pd.DataFrame(user_data, index=[0])
st.write(user_df)
st.write(user_df.shape)

# "Predict" button
if st.sidebar.button("Predict"):
    # Perform prediction or any other action here when the button is clicked
    # You can use the user_data dictionary for prediction
    st.write("## User Input Data Processed")
    df = user_df.copy()
    # Pre-processing Stage
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0})
    df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
    df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, columns=['MultipleLines', 'InternetService', 'OnlineSecurity',
                                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                     'StreamingMovies',
                                     'Contract', 'PaymentMethod'])
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df = df.dropna(axis=0)
    df = df.replace({True: 1, False: 0})
    scaler = StandardScaler()
    df[['MonthlyCharges']] = scaler.fit_transform(df[['MonthlyCharges']])
    df[['TotalCharges']] = scaler.fit_transform(df[['TotalCharges']])
    df[['tenure']] = scaler.fit_transform(df[['tenure']])


    # Define the full list of features
    full_feature_list = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
       'MultipleLines_No', 'MultipleLines_No phone service',
       'MultipleLines_Yes', 'InternetService_DSL',
       'InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_No', 'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes', 'OnlineBackup_No',
       'OnlineBackup_No internet service', 'OnlineBackup_Yes',
       'DeviceProtection_No', 'DeviceProtection_No internet service',
       'DeviceProtection_Yes', 'TechSupport_No',
       'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',
       'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No internet service',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

    # Check for missing columns and add them with a value of 0
    missing_columns = set(full_feature_list) - set(df.columns)
    for column in missing_columns:
        df[column] = 0

    # Define the desired column order
    desired_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'MultipleLines_No',
        'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_DSL',
        'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No',
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No',
        'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No',
        'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No',
        'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',
        'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No',
        'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_Month-to-month',
        'Contract_One year', 'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]

    df = df[desired_order]

    st.write(df.head())
    st.write(df.shape)


    with open('xgb_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    yhat = model.predict(df)

    if yhat[0] == 1:
        result = "Churn"
    else:
        result = "No churn"

    st.write(yhat[0])
    st.write(result)






