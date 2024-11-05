import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
import math
import statsmodels.api as sm
import func as fn


#Setting up pandas display settings
pd.options.mode.copy_on_write = True
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.3f}'.format

#import dataset
raw_dataset=pd.read_csv('2018_03.csv')

#dropping missing observations
cleaned_dataset=raw_dataset.dropna(axis=0,how='any')
cleaned_dataset['date']=pd.to_datetime(cleaned_dataset['date'])
cleaned_dataset['scheduled_time']=pd.to_datetime(cleaned_dataset['scheduled_time'])
cleaned_dataset['actual_time']=pd.to_datetime(cleaned_dataset['actual_time'])
print(cleaned_dataset.head())
target_col=cleaned_dataset['delay_minutes']
std_dataset=fn.standardized(cleaned_dataset)
cleaned_dataset_encoded = pd.get_dummies(cleaned_dataset, columns=['status', 'line', 'type'] , drop_first=True, dtype=int)

# train_df, train_target, test_df, test_target = train_test_split(cleaned_dataset,target_col,test_size=0.2)
train_df, test_df, train_target, test_target = train_test_split(cleaned_dataset_encoded.drop(columns=['delay_minutes']),
                                                                cleaned_dataset['delay_minutes'],
                                                                test_size=0.2,
                                                                random_state=5805)

# Selecting only numeric columns for training
train_df_numeric = train_df.select_dtypes(include=[np.number])
test_df_numeric = test_df.select_dtypes(include=[np.number])

# Dropping target column from the features dataframe
train_df_numeric = train_df_numeric.drop(columns=['delay_minutes'], errors='ignore')
test_df_numeric = test_df_numeric.drop(columns=['delay_minutes'], errors='ignore')

#random forest algorithm
random_forest = RandomForestRegressor(random_state=5805)
random_forest.fit(train_df_numeric, train_target)
feature_importances = pd.Series(random_forest.feature_importances_, index=train_df_numeric.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)
plt.figure(figsize=(12, 8))
feature_importances_sorted.plot(kind='barh')
plt.title('Random Forest - Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.grid(axis='x')
# plt.gca().invert_yaxis()
plt.show()


# selected_features_rf = feature_importances_sorted[feature_importances_sorted > threshold].index.tolist()
# eliminated_features_rf = feature_importances_sorted[feature_importances_sorted <= threshold].index.tolist()

X_train_rf_const = sm.add_constant(train_df_numeric)
ols_model_rf = sm.OLS(train_target, X_train_rf_const).fit()
X_test_rf_const = sm.add_constant(test_df_numeric)
RF_pred_sales = ols_model_rf.predict(X_test_rf_const)
# rf_pred_sales_reverse=inverse_std(RF_pred_sales, test_mean, test_std)
# plt.plot(np.arange(len(test_target)),test_target, label='Original Sales')
# plt.plot(np.arange(len(RF_pred_sales)), RF_pred_sales, label='Predicted Sales')
# plt.title('Random Forest - Original Sales vs Predicted Sales')
# plt.legend(loc='best')
# plt.grid(True)
# plt.show()

mse_rf = round(mean_squared_error(test_target, RF_pred_sales),3)
rf_metrics={
'R-squared': ols_model_rf.rsquared,
'Adjusted R-squared': ols_model_rf.rsquared_adj,
'AIC': ols_model_rf.aic,
'BIC': ols_model_rf.bic,
'MSE': mse_rf
}
# print(rf_metrics)
print(f"\nRandom Forest Metrics\nR-Squared: {round(ols_model_rf.rsquared,3)}\nAdjusted R-Squared: {round(ols_model_rf.rsquared_adj,3)}\nAIC: {round(ols_model_rf.aic,3)},\nBIC: {round(ols_model_rf.bic,3)}\nMean Squared Error: {round(mse_rf,3)}")

train_df, test_df, train_target, test_target = train_test_split(std_dataset.drop(columns=['delay_minutes']),
                                                                std_dataset['delay_minutes'],
                                                                test_size=0.2,
                                                                random_state=5805)

pca = PCA()
# carseat_train_pca = pca.fit_transform(X_dummy)
train_df_pca = pca.fit_transform(train_df_numeric)
explained_variance = np.cumsum(pca.explained_variance_ratio_)


n_components_95 = np.argmax(explained_variance >= 0.95) + 1
print(f"\nNumber of components explaining more than 95% variance: {n_components_95}")

plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.axhline(y=0.95, color='g', linestyle='--')
plt.axvline(x=n_components_95, color='g', linestyle='--')
plt.title('PCA - Cumulative explained variance versus the number of features.')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(np.arange(1,len(explained_variance)+1))
plt.grid(True)
plt.show()

#Standardizing the dataset
# std_dataset=fn.standardized(cleaned_dataset)
# std_dataset=fn.centering_df(cleaned_dataset)
# #Performing Single value decomposition on the dataset
# U,S,Vh=np.linalg.svd(cleaned_dataset)
# print(U,S,Vh)

# covariance_mat=fn.cov_matrix(cleaned_dataset)
# print(covariance_mat)