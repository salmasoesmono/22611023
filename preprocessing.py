import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
data = pd.read_csv('https://raw.githubusercontent.com/salmasoesmono/22611023/main/insurance.csv')

# Periksa nilai hilang
print("Nilai hilang per kolom:")
print(data.isnull().sum())

# Tangani nilai hilang
# Untuk kolom kategorik, ganti nilai hilang dengan modus
categorical_cols = ['sex', 'smoker', 'region']
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Untuk kolom numerik, ganti nilai hilang dengan rata-rata
numeric_cols = ['age', 'bmi', 'children', 'charges']
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].mean())

# Buat objek LabelEncoder
label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Buat objek StandardScaler
scaler = StandardScaler()

# Skala fitur numerik
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Tampilkan tipe data setelah preprocessing
print("\nTipe data setelah preprocessing:")
print(data.dtypes)

