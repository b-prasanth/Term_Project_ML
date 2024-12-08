import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import func as fn
from tabulate import tabulate


# T-test Analysis Function
def t_test_analysis(model):
    print("\nT-test Analysis Results:")
    t_test_results = model.tvalues
    p_values = model.pvalues
    t_test_summary = pd.DataFrame({
        't-value': t_test_results,
        'p-value': p_values
    })
    print(t_test_summary)
    significant_features = t_test_summary[t_test_summary['p-value'] < 0.05]
    print("\nSignificant Features (p-value < 0.05):")
    print(significant_features)

# F-test Analysis Function
def f_test_analysis(model):
    print("\nF-test Analysis Results:")
    f_statistic = model.fvalue
    f_p_value = model.f_pvalue

    print(f"F-statistic: {f_statistic:.4f}")
    print(f"p-value of F-statistic: {f_p_value:.4f}")
    if f_p_value < 0.05:
        print("The regression model is statistically significant.")
    else:
        print("The regression model is not statistically significant.")

# Preprocess and Resample Function
def preprocess_and_resample(df, target_column, time_column="scheduled_time", freq="D"):

    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    df = df.set_index(time_column)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if target_column not in numeric_columns:
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    df = df.dropna()
    df_resampled = df[numeric_columns].resample(freq).mean()
    df_resampled[target_column] = df[target_column].resample(freq).mean()
    df_resampled = df_resampled.dropna()
    return df_resampled


# Backward Stepwise Regression Function
def backward_stepwise_regression(X, y):
    variables = list(X.columns)
    removed_features = []
    metrics_table = []
    selected_features = []

    while len(variables) > 0:
        X_with_const = sm.add_constant(X[variables])
        model = sm.OLS(y, X_with_const).fit()

        # Calculate metrics
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        aic = model.aic
        bic = model.bic
        mse = mean_squared_error(y, model.fittedvalues)
        p_values = model.pvalues.iloc[1:]
        max_p_value = p_values.max()
        feature_to_remove = p_values.idxmax()

        if max_p_value > 0.1:
            metrics_table.append({
                "Feature Removed": feature_to_remove,
                "R-squared": r_squared,
                "Adjusted R-squared": adj_r_squared,
                "AIC": aic,
                "BIC": bic,
                "MSE": mse
            })
            variables.remove(feature_to_remove)
            removed_features.append(feature_to_remove)
        else:
            selected_features=variables.copy()
            break
    metrics_df = pd.DataFrame(metrics_table)
    return model, metrics_df, removed_features, selected_features

def confidence_interval_analysis(model):
    return model.conf_int()


def do_stepwise_regression(df, target_column):
    df_cleaned = preprocess_and_resample(df, target_column)
    X = df_cleaned.drop(columns=[target_column])
    y = df_cleaned[target_column]
    if X.shape[0] < 10:
        raise ValueError("Not enough data after resampling for train-test split")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)
    multi_linear(X_train, X_test, y_train, y_test)
    X_train.drop(columns=['delay_minutes'], inplace=True)
    X_test.drop(columns=['delay_minutes'], inplace=True)
    final_model, metrics_df, removed_features, selected_features = backward_stepwise_regression(X_train, y_train)
    X_test_with_const = sm.add_constant(X_test[final_model.model.exog_names[1:]])
    y_pred = final_model.predict(X_test_with_const)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r_squared = final_model.rsquared
    print("\nBackward Stepwise Regression Results:")
    # print("Final Model Summary:\n", final_model.summary())
    print("\nConfidence Intervals:\n", confidence_interval_analysis(final_model))
    fn.plot_stepwise(y_train, y_test, y_pred)
    t_test_analysis(final_model)
    f_test_analysis(final_model)

    metrics_df = metrics_df.reset_index(drop=True)
    metrics_df.index = metrics_df.index + 1

    metrics_table = tabulate(metrics_df, headers='keys', tablefmt='grid', floatfmt='.4f')
    print("\nMetrics at Each Step of Backward Stepwise Elimination:\n", metrics_table)
    print("\nSelected Features from Backward Stepwise Elimination:\n", selected_features)
    print(f"\nMSE for the final model:\n{test_mse:.4f}" )
    print(f"\nR-squared for final model:\n{test_r_squared:.4f}")


def multi_linear(X_train, X_test, y_train, y_test):
    X_train=sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    # print(X_train.dtypes)
    model = sm.OLS(y_train, X_train).fit()
    y_pred = model.predict(X_test)

    print("\nMultiple Linear Regression:")
    mse_train=mean_squared_error(y_train, model.predict(X_train))
    adj_r_squared = model.rsquared_adj
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("R-squared:", r2)
    print("Adjusted R-squared:", adj_r_squared)
    print("AIC:", model.aic)
    print("BIC:", model.bic)
    print("MSE-Train:", mse_train)
    print("MSE-Test:", mse)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Perfect fit')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()