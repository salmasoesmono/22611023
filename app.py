import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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
X = data.drop(["age", "age_category"], axis=1)  # Features
y = data["age_category"]  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Debugging: Print shapes and sample data
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("First few rows of X_train:\n", X_train.head())
print("First few rows of y_train:\n", y_train.head())

model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Show model performance
st.title('Model Performance')
st.subheader('Classification Report')
report = classification_report(y_test, y_pred, output_dict=True)
st.json(report)

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
