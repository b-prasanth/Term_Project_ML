from datetime import datetime

import exp_data_analysis as eda
import numpy as np
import regression_analysis as ra
import classifier_analysis as cla
import clustering_association as ca
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

#Phase 1 Implementation of the project
#Function for data pre-processing, EDA and Dimensionality reduction
def phase_1():
    print('-' * 50 + " Phase 1 " + '-' * 50)

    data, from_dict, to_dict = eda.load_data()
    data_cleaned = eda.clean_data(data)

    # # Show EDA graphs
    eda.aggregate_data(data_cleaned)
    eda.do_eda_graphs(data_cleaned)

    # Perform encoding and Split into test and train
    data_cleaned = eda.data_encode(data_cleaned)
    target_col = 'arrival_status'

    data_cleaned.drop(columns=['status_estimated'], inplace=True)
    X = data_cleaned.drop(columns=target_col)
    X.drop(columns=['arrival_minutes', 'delay_minutes'], inplace=True)
    y = data_cleaned[target_col]

    y = y.replace({
        'early': 1,
        'on-time': 2,
        'late': 3
    })

    X_copy = data_cleaned.copy()
    # X_copy.drop(columns=['delay_minutes', 'actual_time', 'date', 'arrival_minutes'], inplace=True)
    X_copy.drop(columns=['actual_time', 'date',], inplace=True)
    X_copy = X_copy.replace({
        'morning': 1,
        'afternoon': 2,
        'evening': 3,
        'night': 4
    })

    X_train, X_test, y_train, y_test = eda.func_train_test_split(X, y)

    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # Feature Selection Techniques
    # Random Forest Feature Importance
    selected_features_rf, _ = eda.random_forest_regressor(X_train, X_test, y_train, y_test)

    # PCA
    n_components_95, _ = eda.pca_exp_var(X_train, X_test, y_train, y_test)

    # VIF
    filtered_data_vif, remove_data = eda.do_vif(data_cleaned, 'arrival_status')
    # filtered_data_vif, remove_data = eda.do_vif(data_cleaned.drop(columns=['delay_minutes']), 'arrival_status')
    print("Dropping features with high collinearity",remove_data)
    print(remove_data['Feature'])
    data_cleaned.drop(columns=remove_data['Feature'], inplace=True)

    # SVD
    cumulative_variance, _, top_features_svd = eda.do_svd(data_cleaned)

    #Check for anomaly and remove
    data_cleaned=eda.anomaly_outlier(data_cleaned)

    # Evaluate model performance with each feature selection method
    rf_accuracy = eda.evaluate_model(X_train[selected_features_rf], X_test[selected_features_rf], y_train, y_test)
    pca_accuracy = eda.evaluate_model(X_train.iloc[:, :n_components_95], X_test.iloc[:, :n_components_95], y_train, y_test)
    # vif_accuracy = fs.evaluate_model(X_train[filtered_data_vif['Feature']], X_test[filtered_data_vif['Feature']], y_train, y_test)
    svd_accuracy = eda.evaluate_model(X_train[top_features_svd], X_test[top_features_svd], y_train, y_test)

    # Print out performance for each method
    print(f"Random Forest Feature Selection Accuracy: {rf_accuracy}")
    print(f"PCA Feature Selection Accuracy: {pca_accuracy}")
    # print(f"VIF Feature Selection Accuracy: {vif_accuracy}")
    print(f"SVD Feature Selection Accuracy: {svd_accuracy}")

    # Choose the best feature selection method
    best_method = max([
        ('Random Forest', rf_accuracy),
        ('PCA', pca_accuracy),
        # ('VIF', vif_accuracy),
        ('SVD', svd_accuracy)
    ], key=lambda x: x[1])

    print(f"\nBest Feature Selection Method: {best_method[0]} with Accuracy: {best_method[1]}")

    if best_method[0] == 'Random Forest':
        selected_features = selected_features_rf
    elif best_method[0] == 'PCA':
        selected_features = X_train.columns[:n_components_95]
    elif best_method[0] == 'VIF':
        selected_features = filtered_data_vif['Feature']
    elif best_method[0] == 'SVD':
        selected_features = top_features_svd

    selected_features.append('scheduled_time')
    selected_features.append('arrival_status')
    pre_processed_df = data_cleaned[selected_features]
    # pre_processed_df2=data_cleaned[selected_features_rf]
    class_counts, is_balanced=eda.check_balance(pre_processed_df[target_col])
    balanced_df=pre_processed_df
    if is_balanced is False:
        balanced_df = cla.preprocess_and_balance(pre_processed_df, datetime_col='scheduled_time', target_col='arrival_status', balance_method='oversample')
        class_counts, is_balanced=eda.check_balance(balanced_df[target_col])

    balanced_df.drop(columns=['month', 'day', 'hour', 'day_of_week'], inplace=True)
    pre_processed_df.drop(columns=['month', 'day', 'hour', 'day_of_week'], inplace=True)
    eda.do_eda_profiling(data_cleaned)
    print('-' * 50 + " End of Phase 1 " + '-' * 50)
    return balanced_df, pre_processed_df, data_cleaned

#Phase 2 Implementation for Project
#Function call for Stepwise Regression
def phase_2(data_cleaned):
    # #Backward stepwise regression
    print('-' * 50 + " Phase 2 " + '-' * 50)
    ra.do_stepwise_regression(data_cleaned,'arrival_minutes')
    print('-' * 50 + " End of Phase 2 " + '-' * 50)

#Phase 3 Implementation of the project
#Function call for all the classifiers required for the project
def phase_3(balanced_df, target_col):
    print('-' * 50 + " Phase 3 " + '-' * 50)
    X_resampled=balanced_df.drop(columns=target_col)
    y_resampled=balanced_df[target_col]

    X_train, X_test, y_train, y_test= eda.func_train_test_split(X_resampled, y_resampled)

    print('-' * 50 + " Pre-Pruning Decision Tree " + '-' * 50)
    cla.pre_pruning_dt(X_train, X_test, y_train, y_test)
    print('-' * 50 + " Post-Pruning Decision Tree " + '-' * 50)
    cla.post_pruning_dt(X_train, X_test, y_train, y_test)
    print('-' * 50 + " Logistic Regression " + '-' * 50)
    cla.logistic_regression(X_train, X_test, y_train, y_test)
    print('-' * 50 + " K-Nearest Neighbors " + '-' * 50)
    cla.knn(X_train, X_test, y_train, y_test)
    print('-' * 50 + " Naive Bayes Classifier " + '-' * 50)
    cla.naive_bayes(X_train, X_test, y_train, y_test)
    print('-' * 50 + " Neural Networks - MultiLayered Perceptron " + '-' * 50)
    cla.neural_networks(X_train, X_test, y_train, y_test)
    print('-' * 50 + " Random Forest Classifier " + '-' * 50)
    cla.random_forest(X_train, X_test, y_train, y_test)
    print('-' * 50 + " Random Forest Classifier - Bagging " + '-' * 50)
    cla.random_forest_bagging(X_train, X_test, y_train, y_test)
    print('-' * 50 + " Random Forest Classifier - Boosting " + '-' * 50)
    cla.random_forest_boosting(X_train, X_test, y_train, y_test)
    print('-' * 50 + " Random Forest Classifier - Stacking " + '-' * 50)
    cla.random_forest_stacking(X_train, X_test, y_train, y_test)
    print('-' * 50 + " Support Vector Machine " + '-' * 50)
    cla.svm_classifier(X_train, X_test, y_train, y_test)
    print('-' * 50 + " Metrics from all classifiers " + '-' * 50)
    cla.display_metrics()
    print('-' * 50 + " End of Phase 3 " + '-' * 50)

#Phase 4 Implementation of the project
#Function call for all the clustering algorithms
def phase_4(dataset):
    print('-' * 50 + " Phase 4 " + '-' * 50)
    # ca.kmeans_clustering(dataset)
    # ca.dbscan_clustering(dataset,5)
    ca.apriori_analysis(dataset)
    print('-' * 50 + " End of Phase 4 " + '-' * 50)

#Main Function to run the project
if __name__ == '__main__':
    start=datetime.now()
    start.isoformat(timespec='milliseconds')
    print(f"Project Run start time: {start}")
    balanced_df, pre_processed_df, data_cleaned=phase_1()
    phase_2(data_cleaned)
    phase_3(balanced_df, target_col='arrival_status')
    phase_4(data_cleaned)
    end = datetime.now()
    end.isoformat(timespec='milliseconds')
    print(f"Project Run End time: {end}")
    time_taken = end - start
    print(f"Time taken to run project: {time_taken}")