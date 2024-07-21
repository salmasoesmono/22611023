import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    st.title('Customer Engagement Analysis')

    st.header('Data Exploration')
    data = pd.read_csv('onlinefoods.csv')
    st.write(data.head())

    st.header('Data Preprocessing')
    if st.button('Preprocess Data'):
        # Example code for preprocessing
        data['Monthly Income'] = data['Monthly Income'].replace({
            'No Income': 0,
            'Below Rs.10000': 5000,
            'Rs.10001 - Rs.20000': 15000,
            'Rs.20001 - Rs.30000': 25000,
            'Rs.30001 - Rs.40000': 35000,
            'Rs.40001 - Rs.50000': 45000,
            'More than 50000': 60000,
            '10001 to 25000': 17500
        })
        data = data.dropna(subset=['Monthly Income'])
        st.write('Data Preprocessed', data.head())

    st.header('Model Training')
    if st.button('Train Model'):
        X = data.drop("Feedback", axis=1)
        y = data["Feedback"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        st.write('Model trained successfully!')

if __name__ == "__main__":
    main()
