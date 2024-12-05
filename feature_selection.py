import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    X_scaled = fn.standardized_new(X, ig_cols)
    vif_data['Feature'] = features
    vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    filtered_data = vif_data[vif_data['VIF'] < 5]
    print("\nVIF Number of Features", len(filtered_data))
    print("\nVIF Data:\n", filtered_data)
    return filtered_data

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

    data=data.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
    # Compute Z-Score for each column
    z_scores = (data - data.mean()) / data.std()

    # Set a threshold (e.g., 3 for detecting anomalies)
    threshold = 3

    # Identify anomalies
    anomalies_z = (z_scores.abs() > threshold)

    # Get rows with anomalies
    anomalous_rows = data[anomalies_z.any(axis=1)]
    print("Z-Score Anomalies:\n", anomalous_rows)

    # Compute Q1, Q3, and IQR for each column
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # Define lower and upper bounds for anomalies
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify anomalies
    anomalies_iqr = (data < lower_bound) | (data > upper_bound)

    # Get rows with anomalies
    anomalous_rows_iqr = data[anomalies_iqr.any(axis=1)]
    print("IQR Anomalies:\n", anomalous_rows_iqr)

# def check_balance(X, y):
#     class_counts = y.value_counts()
#     proportions = class_counts / class_counts.sum()
#     # Display counts and proportion of each class
#     print("\nClass Distribution:\n", class_counts)
#
#     if proportions.nunique() == 1:
#         print("Class distribution is equal.")
#         return class_counts, X, y
#     else:
#         pf.display_balance_plot(class_counts)
#         print("Class distribution is not equal.\nBalancing the dataset using SMOTE")
#         X_resampled, y_resampled = perform_balance(X, y)
#         class_counts, X_resampled, y_resampled=check_balance(X_resampled, y_resampled)
#         return class_counts, X_resampled, y_resampled

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