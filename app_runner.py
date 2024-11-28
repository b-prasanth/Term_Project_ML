import plot_func as pf
import feature_selection as fs
import func as fn
import exp_data_analysis as eda
import Data_Preprocessing as dp
import numpy as np

#Load data and Clean data
data, from_dict, to_dict=dp.load_data()
data_cleaned=dp.clean_data(data)

#Show EDA graphs
# eda.aggregate_data(data_cleaned)
# eda.do_eda_graphs(data_cleaned)

#Perform encoding and Split into test and train
data_cleaned=dp.data_encode(data_cleaned)
target_col='arrival_status'
X=data_cleaned.drop(columns=target_col)
X.drop(columns=['delay_minutes','arrival_minutes'], inplace=True)
# X.drop(columns=['arrival_minutes'], inplace=True)
y=data_cleaned[target_col]

y=y.replace({
    'early': 1,
    'on-time': 2,
    'late': 3
})

X_train, X_test, y_train, y_test= fs.func_train_test_split(X, y)
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Implement feature selection algorithms

#Random Forest feature importance and plot
# selected_features_rf, feature_importances_sorted=fs.random_forest_regressor(X_train, X_test, y_train, y_test)
# pf.rf_feature_imp_plot(feature_importances_sorted)
#
# #PCA
# n_components_95, explained_variance =fs.pca_exp_var(X_train, X_test, y_train, y_test)
# pf.pca_exp_var_plot(n_components_95, explained_variance,'PCA')
#
# #VIF
# fs.do_vif(data_cleaned.drop(columns=['delay_minutes']),'arrival_status')
#
# #SVD
# cumulative_variance, explained_variance, top_features=fs.do_svd(data_cleaned)
# pf.plt_svd(cumulative_variance, explained_variance)

# fs.anomaly_outlier(data_cleaned)

#Check if dataset is balanced
# class_counts, X_resampled, y_resampled=fs.check_balance(selected_X, y)
# pf.display_balance_plot(class_counts)