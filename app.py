import streamlit as st
import pandas as pd
import exploration
import preprocessing
import train_and_evaluate

st.title("Insurance Data Analysis and Model Training")

# Load data
data_path = 'insurance.csv'
data = pd.read_csv(data_path)

# Exploration
st.header("Exploration")
if st.checkbox("Show raw data"):
    st.write(data.head())

exploration.main(data)

# Preprocessing
st.header("Preprocessing")
preprocessed_data = preprocessing.main(data)
st.write(preprocessed_data.head())

# Model Training and Evaluation
st.header("Model Training and Evaluation")
train_and_evaluate.main(preprocessed_data)
