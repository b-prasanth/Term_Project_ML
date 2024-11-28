import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plot_func as pf

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

    return df_resampled


# Backward Stepwise Regression Function
def backward_stepwise_regression(X, y):
    variables = list(X.columns)
    removed_features = []
    metrics_table = []

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
            break

    # Create final metrics table
    metrics_df = pd.DataFrame(metrics_table)
    return model, metrics_df, removed_features


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
    final_model, metrics_df, removed_features = backward_stepwise_regression(X_train, y_train)

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
    pf.plot_stepwise(y_train, y_test, y_pred)

    # Display metrics table
    print("\nMetrics at Each Step of Backward Elimination:\n", metrics_df)