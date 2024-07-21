import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Show model performance
st.title('Model Training and Performance')
st.subheader('Classification Report')
report = classification_report(y_test, y_pred, output_dict=True)
st.json(report)

st.subheader('Model Accuracy')
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.2f}')

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
