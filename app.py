import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
@st.cache
def load_data():
    return pd.read_csv('insurance.csv')

data = load_data()

# Preprocess data
def preprocess_data(df):
    # Handle missing values
    categorical_cols = ['sex', 'smoker', 'region']
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    numeric_cols = ['age', 'bmi', 'children', 'charges']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

    # Scale numerical features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

# Preprocess the data
data = preprocess_data(data)

# Define features and target variable
X = data.drop(["age"], axis=1)  # Features
y = data["age"]  # Target variable

# Debugging output
st.write("Unique values in target variable (y):", y.unique())
st.write("Data type of target variable (y):", y.dtype)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42)
try:
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Show model performance
    st.title('Model Training and Performance')
    st.subheader('Regression Metrics')
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f'Mean Squared Error: {mse:.2f}')
    st.write(f'R^2 Score: {r2:.2f}')

except ValueError as e:
    st.error(f"An error occurred: {e}")

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
