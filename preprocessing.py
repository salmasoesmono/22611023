import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
data = pd.read_csv(r'D:\SEMESTER 4\MPML\UAS\onlinefoods.csv')

# Periksa nilai hilang
print("Nilai hilang per kolom:")
print(data.isnull().sum())

# Tangani nilai hilang
# Untuk kolom kategorik, ganti nilai hilang dengan modus
for col in ['Gender', 'Marital Status', 'Occupation', 'Educational Qualifications', 'Feedback']:
    data[col] = data[col].fillna(data[col].mode()[0])

# Mapping untuk kolom Monthly Income
income_mapping = {
    'No Income': 0,
    'Below Rs.10000': 5000,
    'Rs.10001 - Rs.20000': 15000,
    'Rs.20001 - Rs.30000': 25000,
    'Rs.30001 - Rs.40000': 35000,
    'Rs.40001 - Rs.50000': 45000,
    'More than 50000': 60000,
    '10001 to 25000': 17500
}

data['Monthly Income'] = data['Monthly Income'].replace(income_mapping)

# Ubah kolom Monthly Income ke tipe data numerik
data['Monthly Income'] = pd.to_numeric(data['Monthly Income'], errors='coerce')

# Hapus baris dengan nilai yang tidak valid
data = data.dropna(subset=['Monthly Income'])

# Untuk kolom numerik lainnya, ganti nilai hilang dengan rata-rata
for col in ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']:
    data[col] = data[col].fillna(data[col].mean())

# Buat objek LabelEncoder
label_encoder = LabelEncoder()
for col in ['Gender', 'Marital Status', 'Occupation', 'Educational Qualifications', 'Feedback']:
    data[col] = label_encoder.fit_transform(data[col])

# Buat objek StandardScaler
scaler = StandardScaler()

# Skala fitur numerik
numerical_cols = ['Age', 'Monthly Income', 'Family size', 'latitude', 'longitude', 'Pin code']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Tampilkan tipe data setelah preprocessing
print("\nTipe data setelah preprocessing:")
print(data.dtypes)
