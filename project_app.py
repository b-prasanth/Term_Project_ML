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
import tabulate

#Setting up pandas display settings
pd.options.mode.copy_on_write = True
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.3f}'.format

#import dataset
raw_dataset=pd.read_csv('2018_03.csv')
# print(raw_dataset.head())
from_stat_df=raw_dataset[['from', 'from_id']]
to_stat_df=raw_dataset[['to', 'to_id']]
from_comb=from_stat_df.drop_duplicates()
to_comb=to_stat_df.drop_duplicates()
# print(from_comb)
# print(to_comb)


#dropping missing observations
cleaned_dataset=raw_dataset.dropna(axis=0,how='any')
cleaned_dataset['date']=pd.to_datetime(cleaned_dataset['date'])
cleaned_dataset['scheduled_time']=pd.to_datetime(cleaned_dataset['scheduled_time'])
cleaned_dataset['actual_time']=pd.to_datetime(cleaned_dataset['actual_time'])
#downsampling dataset
from_dict = dict(cleaned_dataset[['from', 'from_id']].drop_duplicates().values)
to_dict = dict(cleaned_dataset[['to', 'to_id']].drop_duplicates().values)
cleaned_dataset=cleaned_dataset.drop(columns=['from','to'])
cutoff_date = '2018-03-16'
# cleaned_dataset=cleaned_dataset[cleaned_dataset['scheduled_time']<=cutoff_date]

# print(cleaned_dataset.head())

cleaned_dataset['long_delay'] = cleaned_dataset['delay_minutes'] > 5
cleaned_dataset.groupby('line')['long_delay'].mean().sort_values(ascending=False).plot(kind='bar')
plt.show()

ax = cleaned_dataset.groupby('stop_sequence')["delay_minutes"].mean().plot()
ax.set_ylabel("average delay_minutes")
cleaned_dataset.date = pd.to_datetime(cleaned_dataset.date)
x = cleaned_dataset.groupby('date')['delay_minutes'].mean()
fig, ax = plt.subplots()
fig.set_size_inches(20,8)
fig.autofmt_xdate()
ax.plot(x)
ax.set_ylabel('average delay_minutes')
plt.show()

cleaned_dataset['date'] = pd.to_datetime(cleaned_dataset['date'])
cleaned_dataset['day_of_weeks'] = cleaned_dataset['date'].dt.day_name()
cleaned_dataset['date'] = pd.to_datetime(cleaned_dataset['date'])
cleaned_dataset['day_of_weeks'] = cleaned_dataset['date'].dt.day_name()

week_av_delay = cleaned_dataset.groupby(cleaned_dataset['day_of_weeks'] ,as_index = False )['delay_minutes'].mean()
week_av_delay['num'] = [5,1,6,7,4,2,3]
week_av_delay = cleaned_dataset.groupby(cleaned_dataset['day_of_weeks'] ,as_index = False )['delay_minutes'].mean()
week_av_delay['num'] = [5,1,6,7,4,2,3]
print("\nAverage delay on each day of the week\n",week_av_delay)


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

threshold=0.01
selected_features_rf = feature_importances_sorted[feature_importances_sorted > threshold].index.tolist()
eliminated_features_rf = feature_importances_sorted[feature_importances_sorted <= threshold].index.tolist()
print(f"\nRandom Forest selected features:{selected_features_rf}\nRandom Forest eliminated features:{eliminated_features_rf}")

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

# train_df, test_df, train_target, test_target = train_test_split(std_dataset.drop(columns=['delay_minutes']),
#                                                                 std_dataset['delay_minutes'],
#                                                                 test_size=0.2,
#                                                                 random_state=5805)
#
# # Selecting only numeric columns for training
# train_df_numeric = train_df.select_dtypes(include=[np.number])
# test_df_numeric = test_df.select_dtypes(include=[np.number])
#
# # Dropping target column from the features dataframe
# train_df_numeric = train_df_numeric.drop(columns=['delay_minutes'], errors='ignore')
# test_df_numeric = test_df_numeric.drop(columns=['delay_minutes'], errors='ignore')
std_train_df=fn.standardized(train_df_numeric)
std_test_df=fn.standardized(test_df_numeric)
pca = PCA()
# carseat_train_pca = pca.fit_transform(X_dummy)
# train_df_pca = pca.fit_transform(train_df_numeric)

train_df_pca = pca.fit_transform(std_train_df)
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

# print(cleaned_dataset_encoded.shape)


# # cov_dataset=cleaned_dataset.select_dtypes(include=int)
# cov_dataset=cleaned_dataset_encoded[['delay_minutes','status_departed','status_estimated']]
# print(cov_dataset.head())
# covariance_mat=np.cov(cleaned_dataset_encoded[['delay_minutes','status_departed','status_estimated']])
# covariance_mat=fn.cov_matrix(cleaned_dataset_encoded[['delay_minutes','status_departed','status_estimated']])
# print(covariance_mat)
X_train_const = sm.add_constant(std_train_df)
y_train=list(train_target)
# X_train_const=list(X_train_const)
model = sm.OLS(y_train, X_train_const).fit()

# Performing backward elimination and displaying eliminated features in table
selected_features, elimination_table = fn.backward_elimination(std_train_df, y_train)
stepwise_table=tabulate(elimination_table, headers='keys', tablefmt='grid', floatfmt='.3f')
print(f"\nEliminated features table\n{stepwise_table}\nSelected Features:{selected_features}\nEliminated Features:{elimination_table['Feature Eliminated'].to_list()}")


X_train_final = std_train_df[selected_features]
X_train_final_const = sm.add_constant(X_train_final)
final_step_model = sm.OLS(y_train, X_train_final_const).fit()


print("\nOLS Summary after feature elimination\n",final_step_model.summary())

X_test_final = test_df_numeric[selected_features]
X_test_final_const = sm.add_constant(X_test_final)
pred_sales = final_step_model.predict(X_test_final_const)
# scaler=StandardScaler()
# pred_sales_reverse=fn.inverse_std(pred_sales, test_mean, test_std)
#
# plt.plot(np.arange(len(pred_target)), pred_target, label='Original')
# plt.plot(np.arange(len(pred_sales_reverse)), pred_sales_reverse, label='Predicted')
# plt.title('Backward Stepwise regression - Original vs Predicted Sales')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# #Question 2d
# mse = round(mean_squared_error(pred_target, pred_sales_reverse),3)
# print(f"\nQuestion 2d\nMean Squared Error: {mse}\nEnd of Question 2d")

final_step_model_metrics={
    'R-squared': final_step_model.rsquared,
    'Adjusted R-squared': final_step_model.rsquared_adj,
    'AIC': final_step_model.aic,
    'BIC': final_step_model.bic,
    # 'MSE': mse
}