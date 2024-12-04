import plot_func as pf
import feature_selection as fs
import func as fn
import exp_data_analysis as eda
import test as dp
import numpy as np
import regression_analysis as ra
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import classifier_analysis as cla
import clustering_association as ca
import warnings
import env_config

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def phase_1():
    #Load data and Clean data
    data, from_dict, to_dict=dp.load_data()
    data_cleaned=dp.clean_data(data)

    #Show EDA graphs
    eda.aggregate_data(data_cleaned)
    eda.do_eda_graphs(data_cleaned)

    # Generate eda profiling report
    # eda.do_eda_profiling(data_cleaned.drop(columns=['arrival_minutes']))

    #Perform encoding and Split into test and train
    data_cleaned=dp.data_encode(data_cleaned)
    target_col='arrival_status'

    X=data_cleaned.drop(columns=target_col)
    X.drop(columns=['arrival_minutes','delay_minutes'], inplace=True)
    X_copy=data_cleaned.copy()
    X_copy.drop(columns=['delay_minutes', 'actual_time', 'date', 'arrival_minutes'], inplace=True)
    y=data_cleaned[target_col]

    y=y.replace({
        'early': 1,
        'on-time': 2,
        'late': 3
    })

    X_copy=X_copy.replace({
        'morning':1,
        'afternoon':2,
        'evening':3,
        'night':4
    })


    X_train, X_test, y_train, y_test= fs.func_train_test_split(X, y)
    X_train_bkp=X_train
    X_test_bkp=X_test
    y_train_bkp=y_train
    y_test_bkp=y_test
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # # Implement feature selection algorithms
    #
    # #Random Forest feature importance and plot
    selected_features_rf, feature_importances_sorted=fs.random_forest_regressor(X_train, X_test, y_train, y_test)
    pf.rf_feature_imp_plot(feature_importances_sorted)

    # #PCA
    n_components_95, explained_variance =fs.pca_exp_var(X_train, X_test, y_train, y_test)
    pf.pca_exp_var_plot(n_components_95, explained_variance,'PCA')

    # #VIF
    filtered_data=fs.do_vif(data_cleaned.drop(columns=['delay_minutes']),'arrival_status')

    #
    # #SVD
    cumulative_variance, explained_variance, top_features=fs.do_svd(data_cleaned)
    pf.plt_svd(cumulative_variance, explained_variance)

    #
    # # fs.anomaly_outlier(data_cleaned)
    #
    # # Check if dataset is balanced
    # class_counts, X_resampled, y_resampled=fs.check_balance(selected_X, y)

    selected_features_rf.append('scheduled_time')
    selected_features_rf.append('arrival_status')
    pre_processed_df = data_cleaned[selected_features_rf]
    class_counts, is_balanced=fs.check_balance(pre_processed_df[target_col])
    balanced_df=pre_processed_df
    if is_balanced is False:
        balanced_df = cla.preprocess_and_balance(pre_processed_df, datetime_col='scheduled_time', target_col='arrival_status', balance_method='undersample')
        class_counts, is_balanced=fs.check_balance(balanced_df[target_col])
        # pf.display_balance_plot(class_counts)


    # print(pre_processed_df.head())
    # print(balanced_df.head())
    return balanced_df, pre_processed_df, data_cleaned

#Phase 2 Implementation for

def phase_2(data_cleaned):
    # #Backward stepwise regression
    ra.do_stepwise_regression(data_cleaned, 'arrival_minutes')

def phase_3(balanced_df, target_col):

    X_resampled=balanced_df.drop(columns=target_col)
    y_resampled=balanced_df[target_col]
    y_resampled=y_resampled.replace({
        'early': 1,
        'on-time': 2,
        'late': 3
    })
    X_train, X_test, y_train, y_test= fs.func_train_test_split(X_resampled, y_resampled)

    cla.pre_pruning_dt(X_train, X_test, y_train, y_test)
    cla.post_pruning_dt(X_train, X_test, y_train, y_test)
    cla.logistic_regression(X_train, X_test, y_train, y_test)
    cla.knn(X_train, X_test, y_train, y_test)
    cla.naive_bayes(X_train, X_test, y_train, y_test)
    cla.neural_networks(X_train, X_test, y_train, y_test)
    cla.svm_classifier(X_train, X_test, y_train, y_test)
    cla.display_metrics()

def phase_4(dataset):
    ca.kmeans_clustering(dataset)
    ca.dbscan_clustering(dataset,5)
    # ca.apriori_analysis(dataset)


if __name__ == '__main__':
    balanced_df, pre_processed_df, data_cleaned=phase_1()
    phase_2(data_cleaned)
    phase_3(balanced_df, target_col='arrival_status')
    phase_4(data_cleaned)