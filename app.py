import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache
def load_data():
    return pd.read_csv('insurance.csv')

data = load_data()

# Exploration
st.title('Exploratory Data Analysis')
if st.checkbox('Show raw data'):
    st.write(data.head())

# Plotting
st.subheader('Histograms and Boxplots')
numeric_cols = ['age', 'bmi', 'children', 'charges']
for col in numeric_cols:
    st.subheader(f'Histogram of {col}')
    fig, ax = plt.subplots()
    sns.histplot(data[col], bins=30, ax=ax)
    st.pyplot(fig)
    
    st.subheader(f'Boxplot of {col}')
    fig, ax = plt.subplots()
    sns.boxplot(x=data[col], ax=ax)
    st.pyplot(fig)

st.subheader('Countplots for Categorical Variables')
categorical_cols = ['sex', 'smoker', 'region']
for col in categorical_cols:
    st.subheader(f'Countplot of {col}')
    fig, ax = plt.subplots()
    sns.countplot(x=col, data=data, ax=ax)
    st.pyplot(fig)
