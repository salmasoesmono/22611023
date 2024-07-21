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
    # Tangani nilai hilang
    categorical_cols = ['sex', 'smoker', 'region']
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    numeric_cols = ['age', 'bmi', 'children', 'charges']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Encode variabel kategorikal
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

    # Skala fitur numerik
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

# Preprocess the data
data = preprocess_data(data)

# Define features and target variable
X = data.drop(["age"], axis=1)  # Fitur
y = data["age"]  # Variabel target

# Debugging output
st.write("Nilai unik dalam variabel target (y):", y.unique())
st.write("Tipe data variabel target (y):", y.dtype)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42)
try:
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Show model performance
    st.title('Pelatihan Model dan Kinerja')
    st.subheader('Metrik Regresi')
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f'Kesalahan Kuadrat Rata-rata (MSE): {mse:.2f}')
    st.write(f'Skor R^2: {r2:.2f}')

except ValueError as e:
    st.error(f"Terjadi kesalahan: {e}")

# Plotting
st.subheader('Histogram dan Boxplot')
numeric_cols = ['age', 'bmi', 'children', 'charges']
for col in numeric_cols:
    st.subheader(f'Histogram dari {col}')
    fig, ax = plt.subplots()
    sns.histplot(data[col], bins=30, ax=ax)
    st.pyplot(fig)
    
    st.subheader(f'Boxplot dari {col}')
    fig, ax = plt.subplots()
    sns.boxplot(x=data[col], ax=ax)
    st.pyplot(fig)

st.subheader('Countplot untuk Variabel Kategorikal')
categorical_cols = ['sex', 'smoker', 'region']
for col in categorical_cols:
    st.subheader(f'Countplot dari {col}')
    fig, ax = plt.subplots()
    sns.countplot(x=col, data=data, ax=ax)
    st.pyplot(fig)
