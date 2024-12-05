import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import func as fn
from tabulate import tabulate


# T-test Analysis Function
def t_test_analysis(model):
    print("\nT-test Analysis Results:")
    # Print the t-values and p-values for each coefficient (including the constant)
    t_test_results = model.tvalues
    p_values = model.pvalues
    t_test_summary = pd.DataFrame({
        't-value': t_test_results,
        'p-value': p_values
    })
    print(t_test_summary)

    # Checking which features have a p-value less than 0.05 (significant)
    significant_features = t_test_summary[t_test_summary['p-value'] < 0.05]
    print("\nSignificant Features (p-value < 0.05):")
    print(significant_features)

# F-test Analysis Function
def f_test_analysis(model):
    print("\nF-test Analysis Results:")
    # F-statistic and associated p-value for the overall regression model
    f_statistic = model.fvalue
    f_p_value = model.f_pvalue

    print(f"F-statistic: {f_statistic:.4f}")
    print(f"p-value of F-statistic: {f_p_value:.4f}")

    # If p-value < 0.05, the model is statistically significant
    if f_p_value < 0.05:
        print("The regression model is statistically significant.")
    else:
        print("The regression model is not statistically significant.")

# Preprocess and Resample Function
def preprocess_and_resample(df, target_column, time_column="scheduled_time", freq="D"):
    # Ensure time column is datetime
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    df = df.set_index(time_column)

    # Exclude non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if target_column not in numeric_columns:
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')

    # Drop rows with NaN in any column
    df = df.dropna()

    # Resample numeric columns
    df_resampled = df[numeric_columns].resample(freq).mean()

    # Resample target column
    df_resampled[target_column] = df[target_column].resample(freq).mean()

    # Drop rows with NaN after resampling
    df_resampled = df_resampled.dropna()
    df_resampled.drop(columns=['delay_minutes'], inplace=True)
    #Standardize dataset
    # ig_cols = ['status_departed', 'status_estimated', 'line_Bergen Co. Line', 'line_Gladstone Branch', 'line_Main Line',
    #            'line_Montclair-Boonton', 'line_Morristown Line', 'line_No Jersey Coast', 'line_Northeast Corrdr',
    #            'line_Pascack Valley', 'line_Princeton Shuttle', 'line_Raritan Valley']
    # df_resampled = fn.standardized_new(df_resampled, ig_cols)
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

        # Find the feature with the highest p-value
        p_values = model.pvalues.iloc[1:]  # Exclude constant
        max_p_value = p_values.max()
        feature_to_remove = p_values.idxmax()

        if max_p_value > 0.05:
            # Record only the feature to be removed and the metrics
            metrics_table.append({
                "Feature Removed": feature_to_remove,
                "R-squared": r_squared,
                "Adjusted R-squared": adj_r_squared,
                "AIC": aic,
                "BIC": bic,
                "MSE": mse
            })
            # Remove the feature
            variables.remove(feature_to_remove)
            removed_features.append(feature_to_remove)
        else:
            selected_features=variables.copy()
            break

    # Create final metrics table
    metrics_df = pd.DataFrame(metrics_table)
    return model, metrics_df, removed_features, selected_features


# Confidence Interval Analysis
def confidence_interval_analysis(model):
    return model.conf_int()


def do_stepwise_regression(df, target_column):
    # Preprocess and resample
    df_cleaned = preprocess_and_resample(df, target_column)

    # Split into features and target
    X = df_cleaned.drop(columns=[target_column])
    y = df_cleaned[target_column]

    # Train-test split
    if X.shape[0] < 10:  # Check if there are enough data points after resampling
        raise ValueError("Not enough data after resampling for train-test split")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

    # Backward stepwise regression
    final_model, metrics_df, removed_features, selected_features = backward_stepwise_regression(X_train, y_train)

    # Ensure X_test has only the features in the final model
    X_test_with_const = sm.add_constant(X_test[final_model.model.exog_names[1:]])

    # Evaluate final model on test data
    y_pred = final_model.predict(X_test_with_const)

    # Metrics for the final model
    test_mse = mean_squared_error(y_test, y_pred)
    test_r_squared = final_model.rsquared

    # Print model summary and confidence intervals
    print("Final Model Summary:\n", final_model.summary())
    print("\nConfidence Intervals:\n", confidence_interval_analysis(final_model))

    # Plot Train, Test, and Predictions
    fn.plot_stepwise(y_train, y_test, y_pred)

    # Perform T-test analysis
    t_test_analysis(final_model)

    # Perform F-test analysis
    f_test_analysis(final_model)

    # Display metrics table
    # Reset index and start from 1
    metrics_df = metrics_df.reset_index(drop=True)
    metrics_df.index = metrics_df.index + 1

    metrics_table = tabulate(metrics_df, headers='keys', tablefmt='grid', floatfmt='.4f')
    print("\nMetrics at Each Step of Backward Stepwise Elimination:\n", metrics_table)
    print("\nSelected Features from Backward Stepwise Elimination:\n", selected_features)
    print(f"\nMSE for the final model:\n{test_mse:.4f}" )
    print(f"\nR-squared for final model:\n{test_r_squared:.4f}")
