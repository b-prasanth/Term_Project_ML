from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsOneClassifier

def rf_feature_imp_plot(feature_importances_sorted):
    plt.figure(figsize=(12, 8))
    feature_importances_sorted.plot(kind='barh')
    plt.title('Random Forest - Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.grid(axis='x')
    # plt.gca().invert_yaxis()
    plt.show()

def pca_exp_var_plot(n_components_95, explained_variance, model):
    plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, marker='o')
    plt.axhline(y=0.95, color='g', linestyle='--')
    plt.axvline(x=n_components_95, color='g', linestyle='--')
    plt.title(f'{model} - Cumulative explained variance versus the number of features.')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(np.arange(1, len(explained_variance) + 1))
    plt.grid(True)
    plt.show()

def plt_svd(cumulative_explained_variance,explained_variance):
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
    plt.title("Cumulative Explained Variance by SVD Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.95, '95% variance', color='red', fontsize=12)
    plt.grid()
    plt.show()

def display_balance_plot(class_counts):

    class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Dataset Class distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.xticks([0,1,2], labels=['delayed','on-time','early'])
    plt.show()

def plot_stepwise(y_train, y_test, y_pred):

    plt.figure(figsize=(10, 6))
    plt.plot(y_train.values, label='Train', alpha=0.7)
    plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test.values, label='Test', alpha=0.7)
    plt.plot(range(len(y_train), len(y_train) + len(y_pred)), y_pred, label='Predicted', alpha=0.7, linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Target Variable')
    plt.title('Train, Test, and Predicted Values')
    plt.legend()
    plt.grid()
    plt.show()

#Function to calculate standard deviation.
def calc_std(df_col, df):
    mean = calc_mean(df_col, df)
    var = 0
    for row in df[df_col]:
        var += (row - mean) ** 2
        var = var / len(df)
        std = math.sqrt(var)
    return std

#Function to calculate mean
def calc_mean(df_col,df):
    sum=0
    for row in df[df_col]:
        sum+=row
    return sum/len(df)


def standardized_new(df, cols):
    standardized_dataset=df
    ignor_cols=cols
    numeric_cols=standardized_dataset.select_dtypes(include=['int64',
    'float64','number']).columns
    for cols in numeric_cols:
        if cols not in ignor_cols:
            standardized_dataset[cols]=((standardized_dataset[cols]-
            calc_mean(cols,df))/calc_std(cols,df))
        else:
            continue
    return standardized_dataset

#Function to standardize target col of dataframe
def standardized_target(df):
    standardized_dataset=df
    mean=np.mean(standardized_dataset)
    std=np.std(standardized_dataset)
    standardized_dataset=((standardized_dataset-np.mean(df))/np.std(df))
    return standardized_dataset, mean, std

def calc_accuracy_recall_precision(TP, TN, FP, FN):
    Recall=TP/(TP+FN)
    Precision=TP/(TP+FP)
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    return Recall, Precision, accuracy


def do_smote(DF1):
    smote_oversample = SMOTE(random_state=5805)
    X_Smote, Y_Smote = smote_oversample.fit_resample(DF1.drop(columns='Direction'), DF1['Direction'])

    smote_up = 0
    smote_down = 0
    for i in range(len(Y_Smote)):
        if Y_Smote[i] == 'Up':
            smote_up += 1
        elif Y_Smote[i] == 'Down':
            smote_down += 1

    if smote_up == smote_down:
        print(f"After using SMOTE method\nUp:{smote_up}\nDown:{smote_down}\n\nEnd of Question 4a")
        pd.Series(Y_Smote).value_counts().plot(kind='bar', color=['blue', 'orange'])
        plt.title("Balanced dataset using smote method")
        plt.xticks(rotation=0)
        plt.ylabel('Count')
        plt.xlabel('Direction')
        plt.grid(axis='y')
        plt.show()
        return X_Smote, Y_Smote

def do_lin_reg(xtrain,xtest,ytrain,ytest):
    model = LogisticRegression(random_state=5805)
    model.fit(xtrain, ytrain)
    train_score=model.score(xtrain,ytrain)
    test_score=model.score(xtest,ytest)
    Y_pred = model.predict(xtest)
    y_proba = model.predict_proba(xtest)[:, 1]
    TP, TN, FP, FN=do_roc_auc(ytest,y_proba,Y_pred)
    model_recall, model_precision, model_accuracy = calc_accuracy_recall_precision(TP, TN, FP, FN)
    return model_recall, model_precision, model_accuracy, train_score, test_score

def do_roc_auc(ytest,y_proba,Y_pred):
    FPR, TPR, thresholds = roc_curve(ytest, y_proba)
    AUC = auc(FPR, TPR)
    plt.figure(figsize=(12, 8))
    plt.plot(FPR, TPR, color='blue', linestyle='--', label=f'ROC curve (AUC = {AUC:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    confusion_matrix_op = confusion_matrix(ytest, Y_pred)
    TN = confusion_matrix_op[0, 0]
    FP = confusion_matrix_op[0, 1]
    TP = confusion_matrix_op[1, 1]
    FN = confusion_matrix_op[1, 0]

    conf_mat_plot = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_op)
    conf_mat_plot.plot()
    plt.show()
    return TP, TN, FP, FN


# Method to calculate and plot ROC Curve with AUC
def plot_roc_curve_with_auc(y_test, y_pred_proba, model_name):
    # Binarize the labels for multi-class classification
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)

    # Calculate ROC curve and AUC for the entire model using One-vs-Rest (OvR) strategy
    fpr_all, tpr_all, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
    roc_auc_all = auc(fpr_all, tpr_all)
    # print(f"AUC: {roc_auc_all}")
    # Plot combined ROC curve for the entire model
    plt.figure()
    plt.plot(fpr_all, tpr_all, color='b', label=f'ROC Curve (AUC = {roc_auc_all:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.show()
    return roc_auc_all


# Method to calculate precision, recall, accuracy and plot confusion matrix
def calculate_metrics_and_plot_confusion_matrix(y_true, y_pred, num_classes, type, model_name):

    confusion = confusion_matrix(y_true, y_pred)

    if type == "test":
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion, cmap='Blues', interpolation='nearest')
        plt.title(f'Confusion Matrix for {model_name} test dataset')
        plt.colorbar()
        ticks = np.arange(num_classes)
        plt.xticks(ticks, labels=['early', 'on-time', 'delayed'])
        plt.yticks(ticks, labels=['early', 'on-time', 'delayed'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Display count inside the confusion matrix
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, confusion[i, j], ha="center", va="center", color="black", fontsize=12)
        plt.show()

    precision = []
    recall = []
    f1_score = []
    specificity = []
    accuracy = 0
    total = len(y_true)

    # To store weighted sums
    weighted_precision_sum = 0
    weighted_recall_sum = 0
    weighted_f1_sum = 0
    weighted_specificity_sum = 0
    total_weight = 0

    class_counts = np.sum(confusion, axis=1)

    for i in range(num_classes):
        tp = confusion[i][i]
        fp = sum(confusion[:, i]) - tp
        fn = sum(confusion[i, :]) - tp
        tn = total - (tp + fp + fn)

        # Precision and Recall
        precision_i = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall_i = tp / (tp + fn) if (tp + fn) != 0 else 0

        precision.append(precision_i)
        recall.append(recall_i)

        # F1-Score: harmonic mean of precision and recall
        if (precision_i + recall_i) != 0:
            f1_score_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)
        else:
            f1_score_i = 0
        f1_score.append(f1_score_i)

        # Specificity (True Negative Rate)
        specificity_i = tn / (tn + fp) if (tn + fp) != 0 else 0
        specificity.append(specificity_i)

        # Accuracy
        accuracy += tp

        # Calculate weighted sums
        weight = class_counts[i]  # Number of true instances for this class
        total_weight += weight
        weighted_precision_sum += precision_i * weight
        weighted_recall_sum += recall_i * weight
        weighted_f1_sum += f1_score_i * weight
        weighted_specificity_sum += specificity_i * weight

    # Final accuracy calculation
    accuracy = accuracy / total

    # Calculate weighted averages
    weighted_precision = weighted_precision_sum / total_weight
    weighted_recall = weighted_recall_sum / total_weight
    weighted_f1_score = weighted_f1_sum / total_weight
    weighted_specificity = weighted_specificity_sum / total_weight
    # print(f"Precision: {weighted_precision}")
    # print(f"Recall: {weighted_recall}")
    # print(f"F1-Score: {weighted_f1_score}")
    # print(f"Specificity: {weighted_specificity}")

    return (
        weighted_precision, weighted_recall, accuracy, weighted_f1_score, weighted_specificity, confusion
    )

def plot_roc_curve_with_auc_ovo(X_test, y_test, classifier, model_name):
    # Create One-vs-One classifier
    ovo_clf = OneVsOneClassifier(classifier)
    ovo_clf.fit(X_test, y_test)

    # Since OvO involves pairwise classifiers, we get the ROC curve for each pair
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    n_classes = y_test_bin.shape[1]

    # Store AUC values for each pair of classes
    auc_values = []

    # Iterate over each pair of classes
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            # Extract the probabilities for class i vs class j
            y_test_pair = y_test_bin[:, [i, j]].ravel()
            y_pred_proba_pair = ovo_clf.decision_function(X_test)[:, [i, j]].ravel()

            fpr, tpr, _ = roc_curve(y_test_pair, y_pred_proba_pair)
            roc_auc = auc(fpr, tpr)
            auc_values.append(roc_auc)

            # Plot ROC curve for this pair
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f}) for {i} vs {j}')

    # Plot diagonal line (no discrimination line)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} (OvO)')
    plt.legend(loc='lower right')
    plt.show()

    # Return the average AUC for all pairs
    return np.mean(auc_values)


def evaluate_model(X_train, X_test, y_train, y_test):
    model=LogisticRegression(max_iter=1000, random_state=5805)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score

def rf_feature_importance(X_train,y_train):
    rf=RandomForestClassifier(n_estimators=100, random_state=5805)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    feature_importance_df=pd.DataFrame({
        'Features':X_train.columns,
        'Importance':importances
    }).sort_values(by='Importance', ascending=False)
    return feature_importance_df
