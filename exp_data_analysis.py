import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
# from pandas_profiling import ProfileReport
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
import math
import statsmodels.api as sm
import func as fn
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
from imblearn.over_sampling import SMOTE


#Setting up pandas display settings
pd.options.mode.copy_on_write = True
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.3f}'.format
warnings.filterwarnings('ignore')

def func_train_test_split(X,y):
    train_df, test_df, train_target, test_target = train_test_split(
        X,y,
        test_size=0.2,
        stratify=y,
        random_state=5805)
    return train_df, test_df, train_target, test_target

def random_forest_regressor(X_train, X_test, y_train, y_test):
    random_forest = RandomForestRegressor(random_state=5805)
    random_forest.fit(X_train, y_train)
    feature_importances = pd.Series(random_forest.feature_importances_, index=X_train.columns)
    feature_importances_sorted = feature_importances.sort_values(ascending=False)
    threshold = 0.01
    selected_features_rf = feature_importances_sorted[feature_importances_sorted > threshold].index.tolist()
    eliminated_features_rf = feature_importances_sorted[feature_importances_sorted <= threshold].index.tolist()
    print("\nRandom Forest")
    print(f"\nRandom Forest selected features:{selected_features_rf}\nRandom Forest eliminated features:{eliminated_features_rf}")
    print(f"\nRandom Forest Number of Features selected:{len(selected_features_rf)}\nRandom Forest Number of Features eliminated:{len(eliminated_features_rf)}")
    return selected_features_rf, feature_importances_sorted

def pca_exp_var(X_train, X_test, y_train, y_test):
    ig_cols = ['status_departed', 'status_estimated', 'line_Bergen Co. Line', 'line_Gladstone Branch', 'line_Main Line',
               'line_Montclair-Boonton', 'line_Morristown Line', 'line_No Jersey Coast', 'line_Northeast Corrdr',
               'line_Pascack Valley', 'line_Princeton Shuttle', 'line_Raritan Valley']
    std_train_df = fn.standardized_new(X_train, ig_cols)
    std_test_df = fn.standardized_new(X_test, ig_cols)
    pca = PCA()
    train_df_pca = pca.fit_transform(std_train_df)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(explained_variance >= 0.95) + 1
    print(f"\nPCA\nNumber of components explaining more than 95% variance: {n_components_95}")
    return  n_components_95, explained_variance

def do_vif(cleaned_dataset_encoded,target):
    ig_cols = ['status_departed', 'status_estimated', 'line_Bergen Co. Line', 'line_Gladstone Branch', 'line_Main Line',
               'line_Montclair-Boonton', 'line_Morristown Line', 'line_No Jersey Coast', 'line_Northeast Corrdr',
               'line_Pascack Valley', 'line_Princeton Shuttle', 'line_Raritan Valley']
    vif_data = pd.DataFrame()
    features = cleaned_dataset_encoded.drop(columns=[target]).select_dtypes(include=np.number).columns
    X = cleaned_dataset_encoded[features]
    # X_scaled = fn.standardized_new(X, ig_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    vif_data['Feature'] = features
    vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    filtered_data = vif_data[vif_data['VIF'] < 10]
    remove_data=vif_data[vif_data['VIF'] > 10]
    print("\nVIF Number of Features", len(filtered_data))
    print("\nVIF Data:\n", filtered_data)
    return filtered_data, remove_data

def do_svd(df2):
    # Standardize the data for better performance of SVD
    ig_cols = ['status_departed', 'status_estimated', 'line_Bergen Co. Line', 'line_Gladstone Branch', 'line_Main Line',
               'line_Montclair-Boonton', 'line_Morristown Line', 'line_No Jersey Coast', 'line_Northeast Corrdr',
               'line_Pascack Valley', 'line_Princeton Shuttle', 'line_Raritan Valley']

    numeric_cols = df2.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
    X_numeric = df2[numeric_cols]
    scaler = StandardScaler()
    X_scaled = fn.standardized_new(X_numeric, ig_cols)

    # Perform Singular Value Decomposition
    svd = TruncatedSVD(n_components=min(X_scaled.shape) - 1, random_state=5805)
    svd.fit(X_scaled)
    singular_values = svd.singular_values_
    components = svd.components_

    # Calculate feature importance
    feature_importance = np.abs(components[0])  # First component as example
    feature_ranking = pd.DataFrame({
        'Feature': X_scaled.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    print("\n\nSVD\n")
    print("Feature Ranking:\n", feature_ranking)

    top_features = feature_ranking.head(2)['Feature']
    explained_variance = (singular_values ** 2) / np.sum(singular_values ** 2)
    explained_variance_ratio = svd.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    selected_features = X_scaled.columns[:n_components_95]

    print(f"\nNumber of components explaining more than 95% variance: {n_components_95}")
    print(f"Selected Features: {list(selected_features)}")

    return cumulative_variance, explained_variance, top_features

def anomaly_outlier(data):


    features_for_anomaly = ['delay_minutes', 'long_delay']

    # Step 1: Standardize the features (important for DBSCAN)
    data_anomaly = data[features_for_anomaly].dropna()  # Drop any missing values
    scaler = StandardScaler()
    data_anomaly_scaled = scaler.fit_transform(data_anomaly)

    # Step 2: Apply DBSCAN for anomaly detection
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed
    dbscan_labels = dbscan.fit_predict(data_anomaly_scaled)

    # Step 3: Add DBSCAN labels to the dataset
    data['DBSCAN_Label'] = dbscan_labels
    data['DBSCAN_Outlier'] = dbscan_labels == -1  # Mark outliers with DBSCAN label -1

    # Step 4: Count outliers
    outlier_count = data['DBSCAN_Outlier'].sum()
    print(f"Number of outliers detected by DBSCAN: {outlier_count}")

    # Step 5: Visualization of DBSCAN clustering
    plt.figure(figsize=(8, 6))
    plt.scatter(data['train_id'], data['delay_minutes'], c=dbscan_labels, cmap="viridis", label="Clusters")
    plt.scatter(data[data['DBSCAN_Outlier'] == True]['train_id'],
                data[data['DBSCAN_Outlier'] == True]['delay_minutes'], c="red", label="Outliers")
    plt.title("DBSCAN Clusters and Outliers based on Delay Minutes")
    plt.xlabel("Train ID")
    plt.ylabel("Delay Minutes")
    plt.legend()
    plt.show()

    # Step 6: Remove outliers from the dataset
    data_cleaned = data[~data['DBSCAN_Outlier']]  # Keep only non-outlier rows
    data_cleaned = data_cleaned.drop(columns=["DBSCAN_Label", "DBSCAN_Outlier"])  # Drop helper columns

    print(f"Shape of dataset after removing outliers: {data_cleaned.shape}")
    return data_cleaned


def check_balance(y):

    class_counts = y.value_counts()
    proportions = class_counts / class_counts.sum()
    # Display counts and proportion of each class
    print("\nClass Distribution:\n", class_counts)
    if proportions.nunique() == 1:
        print("Class distribution is equal.")
        fn.display_balance_plot(class_counts)
        return class_counts,True
    else:
        print("\nClass distribution is not equal. Balancing the dataset...")
        fn.display_balance_plot(class_counts)
        return class_counts,False

def perform_balance(X,y):
    smote = SMOTE(random_state=5805)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def evaluate_model(X_train_selected, X_test_selected, y_train, y_test):
    """ Helper function to evaluate a model's performance. You can replace RandomForestClassifier with any model. """
    model = RandomForestClassifier(random_state=5805)
    model.fit(X_train_selected, y_train)
    accuracy = model.score(X_test_selected, y_test)
    return accuracy

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
    # data1 = data
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
    # cleaned_dataset_encoded=data.drop(columns=['status', 'line', 'type'])
    return cleaned_dataset_encoded

def aggregate_data(data_cleaned):
    # Aggregation: Average delay by time of the day
    avg_delay_by_station = data_cleaned.groupby('time_of_day')['delay_minutes'].mean().reset_index()
    print("Average delay by time of the day\n",avg_delay_by_station.sort_values(by='time_of_day', ascending=False))
    # plt.bar(avg_delay_by_station.sort_values(by='time_of_day', ascending=False))
    avg_delay_by_station.sort_values(by='time_of_day', ascending=True).plot(kind='bar')
    plt.grid(axis='y')
    plt.xticks(ticks=[0,1,2,3],labels=["Morning","Afternoon","Evening","Night"], rotation=0)
    plt.xlabel("Time of Day")
    plt.ylabel("Average Delay")
    plt.show()

    # Aggregation: Average delay by date
    avg_delay_by_date = data_cleaned.groupby('date')['delay_minutes'].mean().reset_index()
    print("\nAverage delay by date\n",avg_delay_by_date)
    ticks = list(range(0, 31))
    ticks_labels=list(range(1, 32))
    avg_delay_by_date.sort_values(by='date', ascending=True).plot(kind='bar')
    plt.grid(axis='y')
    plt.xticks(ticks=ticks, labels=ticks_labels)
    plt.xlabel("Date (Mar-2018)")
    plt.ylabel("Average Delay")
    plt.show()

def do_eda_graphs(cleaned_dataset):
    cleaned_dataset['long_delay'] = cleaned_dataset['delay_minutes'] > 5
    cleaned_dataset.groupby('line')['long_delay'].mean().sort_values(ascending=False).plot(kind='bar')
    plt.xticks(rotation=45, fontsize='xx-small')
    plt.grid(axis='y')
    plt.show()

    ax = cleaned_dataset.groupby('stop_sequence')["delay_minutes"].mean().plot()
    ax.set_ylabel("average delay_minutes")
    cleaned_dataset.date = pd.to_datetime(cleaned_dataset.date)
    x = cleaned_dataset.groupby('date')['delay_minutes'].mean()
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)
    fig.autofmt_xdate()
    plt.grid()
    ax.plot(x)
    ax.set_ylabel('average delay_minutes')
    plt.show()

    cleaned_dataset['date'] = pd.to_datetime(cleaned_dataset['date'])
    cleaned_dataset['day_of_weeks'] = cleaned_dataset['date'].dt.day_name()
    cleaned_dataset['date'] = pd.to_datetime(cleaned_dataset['date'])
    cleaned_dataset['day_of_weeks'] = cleaned_dataset['date'].dt.day_name()

    week_av_delay = cleaned_dataset.groupby(cleaned_dataset['day_of_weeks'], as_index=False)['delay_minutes'].mean()
    week_av_delay['num'] = [5, 1, 6, 7, 4, 2, 3]
    week_av_delay = cleaned_dataset.groupby(cleaned_dataset['day_of_weeks'], as_index=False)['delay_minutes'].mean()
    week_av_delay['num'] = [5, 1, 6, 7, 4, 2, 3]
    print("\nAverage delay on each day of the week\n", week_av_delay.sort_values(by='num',ascending=True))

def do_eda_profiling(data_cleaned):
    profile = ProfileReport(
        data_cleaned,
        title="EDA Report for Arrival Delay Analysis",
        explorative=True
    )

    # Save the profiling report
    profile_file = "eda_report.html"
    profile.to_file(profile_file)

    # Create a covariance heatmap using Seaborn
    numeric_data = data_cleaned.select_dtypes(include=[np.number])
    cov_matrix = numeric_data.cov()

    plt.figure(figsize=(12, 10))
    # sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    sns.heatmap(
        cov_matrix,
        annot=False,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={'label': 'Covariance Values'},
        xticklabels=cov_matrix.columns,
        yticklabels=cov_matrix.index,
        mask=np.eye(len(cov_matrix))
    )
    plt.tight_layout()
    plt.title("Covariance Heatmap")

    # Save the heatmap to a base64-encoded image
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    encoded_heatmap = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()

    # Inject the heatmap into the HTML report
    heatmap_html = f"""
    <h2 style="text-align: center;">Covariance Heatmap</h2>
    <div style="text-align: center;">
        <img src="data:image/png;base64,{encoded_heatmap}" alt="Covariance Heatmap" />
    </div>
    """

    # Add the heatmap HTML content to the report
    with open(profile_file, "r+") as eda_file:
        content = eda_file.read()
        # Insert the heatmap HTML content before the end of the report
        updated_content = content.replace("</body>", heatmap_html + "</body>")
        eda_file.seek(0)
        eda_file.write(updated_content)
        eda_file.truncate()