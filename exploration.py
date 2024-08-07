import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Muat dataset (pastikan path file Anda disesuaikan)
data = pd.read_csv(r'D:\SEMESTER 4\MPML\UAS\insurance.csv')

# Inspeksi Data
print(data.head())
print(data.info())  # Periksa tipe data dan nilai hilang
print(data.describe())  # Statistik deskriptif untuk kolom numerik

# Variabel numerik
numeric_cols = ['age', 'bmi', 'children', 'charges']
for col in numeric_cols:
    sns.histplot(data[col], bins=30)
    plt.title(f'Histogram of {col}')
    plt.show()
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Variabel kategorik
categorical_cols = ['sex', 'smoker', 'region']
for col in categorical_cols:
    sns.countplot(x=col, data=data)
    plt.title(f'Countplot of {col}')
    plt.show()
