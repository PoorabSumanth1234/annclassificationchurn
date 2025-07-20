import pandas as pd 
import streamlit as st 
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import pickle 

#Load the trained model 
model = tf.keras.models.load_model('model.h5')

#Load the encoders, scalers 
with open('label_encoder_gender.pkl','rb')as file:
    label_encoder_gen=pickle.load(file)
    
with open('one_hot_encoder_geo.pkl','rb')as file:
    one_hot_encod_geo=pickle.load(file)

with open('scaler.pkl','rb')as file:
    scaler=pickle.load(file)
#'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
       #'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary    
# Streamlit app 
st.title("Customer Churn prediction")

# INput Data 
geography=st.selectbox('Geography',one_hot_encod_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gen.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider("Number Of products",1,4)
has_cr_card=st.selectbox("Has Credit Card",[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gen.transform([gender])[0]],  # FIXED
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# One_Hot_encode geography 
geo_encoded=one_hot_encod_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_encod_geo.get_feature_names_out(['Geography']))

#combine one hot encoded file with input data 
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_data_scaled=scaler.transform(input_data)

#Predict churn 
prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]
st.write(f" Churn Probablity {prediction_proba:.2f}")
if prediction_proba>0.5:
    print("Customer is likely to churn ")
else:
    print("Customer is not likely to churn ")