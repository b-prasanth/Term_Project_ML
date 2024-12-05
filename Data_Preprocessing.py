import pandas as pd


def load_data():
    data = pd.read_csv('2018_03.csv')

    # print(data.dtypes)
    # Convert time columns to datetime
    data['scheduled_time'] = pd.to_datetime(data['scheduled_time'], errors='coerce')
    data['actual_time'] = pd.to_datetime(data['actual_time'], errors='coerce')
    data['arrival_minutes'] = (data['actual_time'] - data['scheduled_time']).dt.total_seconds() / 60
    from_dict = dict(data[['from', 'from_id']].drop_duplicates().values)
    to_dict = dict(data[['to', 'to_id']].drop_duplicates().values)


    agg_data=result = data.groupby(["date", "train_id"]).agg(from_id=("from_id", "first"),to_id=("to_id", "last"),  # Last stop's `to_id`
    scheduled_time=("scheduled_time", "last"),  # Scheduled time of the last stop
    actual_time=("actual_time", "last") , # Actual time of the last stop
                                                             stop_sequence=("stop_sequence", "last"),
                                                             delay_minutes=("delay_minutes", "last"),
                                                             status=("status", "last"),line=("line", "last"),
                                                             type=("type", "last"), arrival_minutes=("arrival_minutes","last")).reset_index()

    # print(agg_data.head())
    # print(len(agg_data))
    data=agg_data


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
    # data.drop(columns=['delay_minutes'], inplace=True)
    # cleaned_dataset = data.drop(columns=['from', 'to'])
    # Drop rows with NaN values
    data_cleaned = data.dropna()

    # Check for duplicates
    duplicates = data_cleaned.duplicated()
    print(f"Number of duplicate rows: {duplicates.sum()}")

    # Drop duplicates
    data_cleaned = data_cleaned.drop_duplicates()
    print(data_cleaned.head())
    return data_cleaned


def data_encode(data):
    cleaned_dataset_encoded = pd.get_dummies(data,
                                             columns=['status', 'line', 'type'],
                                             drop_first=True,
                                             dtype=int)

    # label_encoder = LabelEncoder()
    # cleaned_dataset_encoded=data
    # cleaned_dataset_encoded['status'] = label_encoder.fit_transform(cleaned_dataset_encoded['status'])  # Encode target variable
    # cleaned_dataset_encoded['line'] = label_encoder.fit_transform(cleaned_dataset_encoded['line'])  # Encode 'line' column
    return cleaned_dataset_encoded