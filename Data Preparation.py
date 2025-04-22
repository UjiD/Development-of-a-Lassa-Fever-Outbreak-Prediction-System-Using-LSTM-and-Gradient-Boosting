import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load datasets
outbreak_data = pd.read_csv('lassa_outbreak_history.csv')
environmental_data = pd.read_csv('environmental_factors.csv')
clinical_data = pd.read_csv('clinical_records.csv')

# Merge datasets
merged_data = pd.merge(outbreak_data, environmental_data, on=['region', 'date'])
merged_data = pd.merge(merged_data, clinical_data, on=['region', 'date'])

# Handle missing values
merged_data.fillna(method='ffill', inplace=True)

# Convert date to datetime and set as index
merged_data['date'] = pd.to_datetime(merged_data['date'])
merged_data.set_index('date', inplace=True)

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_data.select_dtypes(include=[np.number]))