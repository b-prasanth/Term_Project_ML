from datetime import datetime

import feature_selection as fs
import func as fn
import exp_data_analysis as eda
import Data_Preprocessing as dp
import numpy as np
import regression_analysis as ra
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import classifier_analysis as cla
import clustering_association as ca
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

#Phase 1 Implementation of the project
#Function EDA and Dimensionality reduction
def phase_1():
    # Load data and Clean data
    print('-' * 50 + " Phase 1 " + '-' * 50)

    data, from_dict, to_dict = dp.load_data()
    data_cleaned = dp.clean_data(data)

    # Show EDA graphs
    eda.aggregate_data(data_cleaned)
    eda.do_eda_graphs(data_cleaned)

    # Perform encoding and Split into test and train
    data_cleaned = dp.data_encode(data_cleaned)
    target_col = 'arrival_status'

    X = data_cleaned.drop(columns=target_col)
    X.drop(columns=['arrival_minutes', 'delay_minutes'], inplace=True)
    y = data_cleaned[target_col]

    y = y.replace({
        'early': 1,
        'on-time': 2,
        'late': 3
    })

    X_copy = data_cleaned.copy()
    X_copy.drop(columns=['delay_minutes', 'actual_time', 'date', 'arrival_minutes'], inplace=True)
    X_copy = X_copy.replace({
        'morning': 1,
        'afternoon': 2,
        'evening': 3,
        'night': 4
    })

    X_train, X_test, y_train, y_test = fs.func_train_test_split(X, y)

    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # Feature Selection Techniques
    # Random Forest Feature Importance
    selected_features_rf, _ = fs.random_forest_regressor(X_train, X_test, y_train, y_test)

    # PCA
    n_components_95, _ = fs.pca_exp_var(X_train, X_test, y_train, y_test)

    # VIF
    filtered_data_vif = fs.do_vif(data_cleaned.drop(columns=['delay_minutes']), 'arrival_status')

    # SVD
    cumulative_variance, _, top_features_svd = fs.do_svd(data_cleaned)

    # Evaluate model performance with each feature selection method
    rf_accuracy = fs.evaluate_model(X_train[selected_features_rf], X_test[selected_features_rf], y_train, y_test)
    pca_accuracy = fs.evaluate_model(X_train.iloc[:, :n_components_95], X_test.iloc[:, :n_components_95], y_train, y_test)
    # vif_accuracy = fs.evaluate_model(X_train[filtered_data_vif['Feature']], X_test[filtered_data_vif['Feature']], y_train, y_test)
    svd_accuracy = fs.evaluate_model(X_train[top_features_svd], X_test[top_features_svd], y_train, y_test)

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

    selected_features_rf.append('scheduled_time')
    selected_features_rf.append('arrival_status')
    pre_processed_df = data_cleaned[selected_features]
    class_counts, is_balanced=fs.check_balance(pre_processed_df[target_col])
    balanced_df=pre_processed_df
    if is_balanced is False:
        balanced_df = cla.preprocess_and_balance(pre_processed_df, datetime_col='scheduled_time', target_col='arrival_status', balance_method='oversample')
        class_counts, is_balanced=fs.check_balance(balanced_df[target_col])
            # pf.display_balance_plot(class_counts)


        # print(pre_processed_df.head())
        # print(balanced_df.head())
    balanced_df.drop(columns=['month', 'day', 'hour', 'day_of_week'], inplace=True)
    print('-' * 50 + " End of Phase 1 " + '-' * 50)
    return balanced_df, pre_processed_df, data_cleaned


# def phase_1():
#     #Load data and Clean data
#     print('-' * 50 + " Phase 1 " + '-' * 50)
#
#     data, from_dict, to_dict=dp.load_data()
#     data_cleaned=dp.clean_data(data)
#
#     #Show EDA graphs
#     eda.aggregate_data(data_cleaned)
#     eda.do_eda_graphs(data_cleaned)
#
#     # Generate eda profiling report
#     # eda.do_eda_profiling(data_cleaned.drop(columns=['arrival_minutes']))
#
#     #Perform encoding and Split into test and train
#     data_cleaned=dp.data_encode(data_cleaned)
#     target_col='arrival_status'
#
#     X=data_cleaned.drop(columns=target_col)
#     X.drop(columns=['arrival_minutes','delay_minutes'], inplace=True)
#     X_copy=data_cleaned.copy()
#     X_copy.drop(columns=['delay_minutes', 'actual_time', 'date', 'arrival_minutes'], inplace=True)
#     y=data_cleaned[target_col]
#
#     y=y.replace({
#         'early': 1,
#         'on-time': 2,
#         'late': 3
#     })
#
#     X_copy=X_copy.replace({
#         'morning':1,
#         'afternoon':2,
#         'evening':3,
#         'night':4
#     })
#
#
#     X_train, X_test, y_train, y_test= fs.func_train_test_split(X, y)
#     X_train_bkp=X_train
#     X_test_bkp=X_test
#     y_train_bkp=y_train
#     y_test_bkp=y_test
#     X_train = X_train.select_dtypes(include=[np.number])
#     X_test = X_test.select_dtypes(include=[np.number])
#
#     # # Implement feature selection algorithms
#     #
#     # #Random Forest feature importance and plot
#     selected_features_rf, feature_importances_sorted=fs.random_forest_regressor(X_train, X_test, y_train, y_test)
#     pf.rf_feature_imp_plot(feature_importances_sorted)
#
#     # #PCA
#     n_components_95, explained_variance =fs.pca_exp_var(X_train, X_test, y_train, y_test)
#     pf.pca_exp_var_plot(n_components_95, explained_variance,'PCA')
#
#     # #VIF
#     filtered_data=fs.do_vif(data_cleaned.drop(columns=['delay_minutes']),'arrival_status')
#
#     #
#     # #SVD
#     cumulative_variance, explained_variance, top_features=fs.do_svd(data_cleaned)
#     pf.plt_svd(cumulative_variance, explained_variance)
#
#     #
#     # # fs.anomaly_outlier(data_cleaned)
#     #
#     # # Check if dataset is balanced
#     # class_counts, X_resampled, y_resampled=fs.check_balance(selected_X, y)
#
#     selected_features_rf.append('scheduled_time')
#     selected_features_rf.append('arrival_status')
#     pre_processed_df = data_cleaned[selected_features_rf]
#     class_counts, is_balanced=fs.check_balance(pre_processed_df[target_col])
#     balanced_df=pre_processed_df
#     if is_balanced is False:
#         balanced_df = cla.preprocess_and_balance(pre_processed_df, datetime_col='scheduled_time', target_col='arrival_status', balance_method='undersample')
#         class_counts, is_balanced=fs.check_balance(balanced_df[target_col])
#         # pf.display_balance_plot(class_counts)
#
#
#     # print(pre_processed_df.head())
#     # print(balanced_df.head())
#     print('-' * 50 + " End of Phase 1 " + '-' * 50)
#     return balanced_df, pre_processed_df, data_cleaned

#Phase 2 Implementation for Project
#Function call for Stepwise Regression
def phase_2(data_cleaned):
    # #Backward stepwise regression
    print('-' * 50 + " Phase 2 " + '-' * 50)
    ra.do_stepwise_regression(data_cleaned, 'arrival_minutes')
    print('-' * 50 + " End of Phase 2 " + '-' * 50)

#Phase 3 Implementation of the project
#Function call for all the classifiers required for the project
def phase_3(balanced_df, target_col):
    print('-' * 50 + " Phase 3 " + '-' * 50)
    X_resampled=balanced_df.drop(columns=target_col)
    y_resampled=balanced_df[target_col]
    y_resampled=y_resampled.replace({
        'early': 1,
        'on-time': 2,
        'late': 3
    })
    X_train, X_test, y_train, y_test= fs.func_train_test_split(X_resampled, y_resampled)

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
    # print('-' * 50 + " Support Vector Machine " + '-' * 50)
    # cla.run_svm(X_train, X_test, y_train, y_test)
    # cla.run_svm_with_grid_search(X_train, X_test, y_train, y_test)
    # cla.svm_classifier(X_train, X_test, y_train, y_test)
    print('-' * 50 + " Metrics from all classifiers " + '-' * 50)
    cla.display_metrics()
    print('-' * 50 + " End of Phase 3 " + '-' * 50)

#Phase 4 Implementation of the project
#Function call for all the clustering algorithms
def phase_4(dataset):
    print('-' * 50 + " Phase 4 " + '-' * 50)
    ca.kmeans_clustering(dataset)
    ca.dbscan_clustering(dataset,5)
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