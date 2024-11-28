import pandas as pd
import numpy as np

# Load dataset

def load_data():
    data = pd.read_csv('2018_03.csv')

    # Convert time columns to datetime
    data['scheduled_time'] = pd.to_datetime(data['scheduled_time'], errors='coerce')
    data['actual_time'] = pd.to_datetime(data['actual_time'], errors='coerce')
    data['arrival_minutes'] = (data['actual_time'] - data['scheduled_time']).dt.total_seconds() / 60
    from_dict = dict(data[['from', 'from_id']].drop_duplicates().values)
    to_dict = dict(data[['to', 'to_id']].drop_duplicates().values)
    return data, from_dict, to_dict

# Create 'time_of_day'
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

# Create 'arrival_status'
def get_arrival_status(delay_minutes):
    if delay_minutes < -0.5:
        return 'early'
    elif -0.5 <= delay_minutes <= 0.5:
        return 'on-time'
    else:
        return 'late'

def clean_data(data):

    data['arrival_status'] = data['arrival_minutes'].apply(get_arrival_status)
    data['time_of_day'] = data['scheduled_time'].dt.hour.apply(get_time_of_day)
    cleaned_dataset = data.drop(columns=['from', 'to'])
    # Drop rows with NaN values
    data_cleaned = cleaned_dataset.dropna()

    # Check for duplicates
    duplicates = data_cleaned.duplicated()
    print(f"Number of duplicate rows: {duplicates.sum()}")

    # Drop duplicates
    data_cleaned = data_cleaned.drop_duplicates()
    return data_cleaned

def data_encode(data):
    cleaned_dataset_encoded = pd.get_dummies(data,
                                             columns=['status', 'line', 'type'],
                                             drop_first=True,
                                             dtype=int)
    return cleaned_dataset_encoded