from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import func as fn
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed  # For parallel execution
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import pandas as pd
from tabulate import tabulate
from sklearn.svm import SVC
from sklearn.utils import resample
import seaborn as sns

warnings.filterwarnings('ignore')

#List to store metrics of each classifier
metrics=[]
cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)
#Function to display the metrics in a
def display_metrics():
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.reset_index(drop=True)
    metrics_df.index = metrics_df.index + 1

    selected_columns = [
        'model', 'Training Accuracy', 'Test Accuracy',
        'Precision', 'Recall',  'F1 Score', 'Specificity','AUC', 'Confusion Matrix'
    ]
    filtered_metrics = [{key: value for key, value in metric.items() if key in selected_columns} for metric in metrics]

    metrics_table = tabulate(filtered_metrics, headers='keys', tablefmt='grid', floatfmt='.3f')
    print(metrics_table)

def display_metrics_console(train_accuracy, test_accuracy, test_precision, test_recall, test_f1, test_specificity, test_auc, confusion_matrix):
    print(f'\nTrain Accuracy: {train_accuracy:.3f}')
    print(f'Test Accuracy: {test_accuracy:.3f}')
    print(f'Precision: {test_precision:.3f}')
    print(f'Recall: {test_recall:.3f}')
    print(f'F1 Score: {test_f1:.3f}')
    print(f'Specificity: {test_specificity:.3f}')
    print(f'AUC: {test_auc:.3f}')
    print(f'Confusion Matrix\n', confusion_matrix)

#Function to perform pre-pruning decision tree
def pre_pruning_dt(X_train, X_test, y_train, y_test):

    start_time = datetime.now()
    print("\nDT Pre-pruning Start-time:", start_time)

    # Pre-pruning parameters
    pre_prune_params = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20],
        'max_features': [None, 'auto', 'sqrt', 'log2']
    }

    dt_pre_prune = DecisionTreeClassifier(random_state=5805)
    grid_search_pre_prune = GridSearchCV(dt_pre_prune, pre_prune_params, cv=cv, n_jobs=-1, verbose=1)
    grid_search_pre_prune.fit(X_train, y_train)
    best_dt_pre_prune = grid_search_pre_prune.best_estimator_
    train_accuracy_pre_prune = accuracy_score(y_train, best_dt_pre_prune.predict(X_train))
    test_accuracy_pre_prune = accuracy_score(y_test, best_dt_pre_prune.predict(X_test))

    # print(f"\nPre-Pruning - Train Accuracy: {train_accuracy_pre_prune:.3f}")
    # print(f"Pre-Pruning - Test Accuracy: {test_accuracy_pre_prune:.3f}")
    print(f"\nBest Parameters: {grid_search_pre_prune.best_params_}")
    print(f"Best Estimators: {best_dt_pre_prune}")

    y_train_pred_pre_prune = best_dt_pre_prune.predict(X_train)
    y_test_pred_pre_prune = best_dt_pre_prune.predict(X_test)
    y_pred_proba_pre_prune = best_dt_pre_prune.predict_proba(X_test)

    # Calculate metrics and plot confusion matrices for Pre-Pruning and Post-Pruning
    train_precision_pre, train_recall_pre, train_accuracy_pre, train_specificity, train_f1_score,train_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_train, y_train_pred_pre_prune, len(np.unique(y_train)), 'train', 'pre-pruning DT')
    test_precision_pre, test_recall_pre, test_accuracy_pre, test_specificity, test_f1_score,test_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_test, y_test_pred_pre_prune, len(np.unique(y_test)), 'test', 'pre-pruning DT')

    auc = fn.plot_roc_curve_with_auc(y_test, y_pred_proba_pre_prune, 'pre-pruning DT')
    auc_ovo=fn.plot_roc_curve_with_auc_ovo(X_test,y_test, best_dt_pre_prune, 'pre-pruning DT')

    display_metrics_console(train_accuracy_pre_prune, test_accuracy_pre_prune, test_precision_pre, test_recall_pre,test_f1_score,test_specificity,auc,test_conf_mat)

    metrics.append({
        'model': 'DT Pre-Pruning',
        'Training Accuracy': train_accuracy_pre,
        'Test Accuracy': test_accuracy_pre,
        'Training Precision': train_precision_pre,
        'Precision': test_recall_pre,
        'AUC': auc,
        'Training Recall': train_recall_pre,
        'Recall': test_recall_pre,
        'Training Specificity': train_specificity,
        'Specificity': test_specificity,
        'Training F1 Score': train_f1_score,
        'F1 Score': test_f1_score,
        'Confusion Matrix': test_conf_mat
    })

    # print(f"\nBelow are the metrics for DT pre-pruning:\n{metrics}")
    end_time = datetime.now()
    print("\nDT Pre-pruning end-time:", end_time)
    print("\nDT Pre-pruning time taken:", end_time - start_time)

    return metrics



def post_pruning_dt(X_train, X_test, y_train, y_test):
    start_time = datetime.now()
    print("\nDT Post-pruning Start-time:", start_time)
    decision_tree = DecisionTreeClassifier(random_state=5805)
    path = decision_tree.cost_complexity_pruning_path(X_train, y_train)
    alphas, impurities = path.ccp_alphas, path.impurities
    train_accuracies = []
    test_accuracies = []

    for alpha in alphas:
        decTreeModel = DecisionTreeClassifier(random_state=5805, ccp_alpha=alpha)
        decTreeModel.fit(X_train, y_train)
        target_train_pred = decTreeModel.predict(X_train)
        target_test_pred = decTreeModel.predict(X_test)
        train_accuracies.append(accuracy_score(y_train, target_train_pred))
        test_accuracies.append(accuracy_score(y_test, target_test_pred))


    # Find the best alpha (alpha with the highest test accuracy)
    best_alpha_idx = test_accuracies.index(max(test_accuracies))
    best_alpha = alphas[best_alpha_idx]
    bestdecTreeModel = DecisionTreeClassifier(random_state=5805, ccp_alpha=best_alpha)
    bestdecTreeModel.fit(X_train, y_train)
    y_train_pred_post_prune = bestdecTreeModel.predict(X_train)
    y_test_pred_post_prune = bestdecTreeModel.predict(X_test)

    y_pred_proba_post_prune = bestdecTreeModel.predict_proba(X_test)

    train_accuracy_post_prune = accuracy_score(y_train, bestdecTreeModel.predict(X_train))
    test_accuracy_post_prune = accuracy_score(y_test, bestdecTreeModel.predict(X_test))

    # print(f"\nPost-Pruning - Train Accuracy: {train_accuracy_post_prune:.3f}")
    # print(f"Post-Pruning - Test Accuracy: {test_accuracy_post_prune:.3f}")
    # print(f"\nBest Parameters: {bestdecTreeModel.best_params_}")

    train_precision_post, train_recall_post, train_accuracy_post, train_specificity, train_f1_score, train_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_train, y_train_pred_post_prune, len(np.unique(y_train)), 'train', 'post-pruning DT')
    test_precision_post, test_recall_post, test_accuracy_post, test_specificity, test_f1_score, test_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_test, y_test_pred_post_prune, len(np.unique(y_test)), 'test', 'post-pruning DT')

    auc = fn.plot_roc_curve_with_auc(y_test, y_pred_proba_post_prune, 'post-pruning DT')

    auc_ovo=fn.plot_roc_curve_with_auc_ovo(X_test,y_test, bestdecTreeModel, 'post-pruning DT')

    display_metrics_console(train_accuracy_post_prune, test_accuracy_post_prune, test_precision_post, test_recall_post, test_f1_score, test_specificity, auc, test_conf_mat)


    metrics.append({
        'model': 'DT Post-Pruning',
        'Training Accuracy': train_accuracy_post,
        'Test Accuracy': test_accuracy_post,
        'Training Precision': train_precision_post,
        'Precision': test_recall_post,
        'AUC': auc,
        'Training Recall': train_recall_post,
        'Recall': test_recall_post,
        'Training Specificity': train_specificity,
        'Specificity': test_specificity,
        'Training F1 Score': train_f1_score,
        'F1 Score': test_f1_score,
        'Confusion Matrix': test_conf_mat
    })

    # print(f"\nBelow are the metrics for DT Post-pruning:\n{metrics}")
    end_time = datetime.now()
    print("\nDT Post-pruning end-time:", end_time)
    print("\nDT Post-pruning time taken:", end_time - start_time)

    return metrics


#Function to perform Logistic Regression
def logistic_regression(X_train, X_test, y_train, y_test):
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    start_time = datetime.now()
    print("\nLogistic Regression Grid Search Start-time:", start_time)

    param_grid = {
        'penalty': ['l2', 'none'],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'max_iter': [500, 1000, 2000, 3000],
    }
    log_reg = LogisticRegression(random_state=5805)
    grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    grid_search.fit(X_train, y_train)
    best_log_reg = grid_search.best_estimator_

    # Train and Test Accuracy for Logistic Regression
    train_accuracy = accuracy_score(y_train, best_log_reg.predict(X_train))
    test_accuracy = accuracy_score(y_test, best_log_reg.predict(X_test))
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Estimator: {best_log_reg}")

    # print(f"\nLogistic Regression - Train Accuracy: {train_accuracy:.3f}")
    # print(f"Logistic Regression - Test Accuracy: {test_accuracy:.3f}")

    y_train_pred = best_log_reg.predict(X_train)
    y_test_pred = best_log_reg.predict(X_test)
    y_pred_proba = best_log_reg.predict_proba(X_test)

    # Calculate Metrics and Plot Confusion Matrix
    train_precision, train_recall, train_accuracy, train_specificity, train_f1_score,train_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_train, y_train_pred, len(np.unique(y_train)), 'train', 'Logistic Regression')
    test_precision, test_recall, test_accuracy, test_specificity, test_f1_score,test_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_test, y_test_pred, len(np.unique(y_test)), 'test', 'Logistic Regression')

    auc = fn.plot_roc_curve_with_auc(y_test, y_pred_proba, 'Logistic Regression')
    auc_ovo = fn.plot_roc_curve_with_auc_ovo(X_test, y_test, best_log_reg, 'Logistic Regression')

    display_metrics_console(train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, test_specificity, auc, test_conf_mat)

    # print(f"\nBelow are the metrics for Logistic Regression Grid Search:\n")
    metrics.append( {
        'model': 'Logistic Regression Grid Search',
        'Training Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Precision': test_recall,
        'AUC': auc,
        'Training Recall': train_recall,
        'Recall': test_recall,
        'Training Specificity': train_specificity,
        'Specificity': test_specificity,
        'Training F1 Score': train_f1_score,
        'F1 Score': test_f1_score,
        'Confusion Matrix': test_conf_mat
    })

    # print(metrics)
    end_time = datetime.now()
    print("\nLogistic Regression Grid Search end-time:", end_time)
    print("\nLogistic Regression Grid Search time taken:", end_time - start_time)

    return metrics

#Function to perform K-Nearest Neighbours
def knn(X_train, X_test, y_train, y_test):

    print("\nFinding Optimal k...")
    k_range = range(1, 31)
    k_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
        k_scores.append(scores.mean())

    # Plot k vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, k_scores, marker='o', linestyle='--', color='b')
    plt.title('Accuracy vs k in KNN')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-Validated Accuracy')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

    print("Optimal k:", k_range[np.argmax(k_scores)])
    print("Corresponding Accuracy:", max(k_scores))
    start_time = datetime.now()
    print("\nKNN Grid Search Start-time:", start_time)
    param_grid = {
        'n_neighbors': [3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2],
        'leaf_size': [20, 30, 40],
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_knn = grid_search.best_estimator_
    train_accuracy = accuracy_score(y_train, best_knn.predict(X_train))
    test_accuracy = accuracy_score(y_test, best_knn.predict(X_test))

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Estimator: {best_knn}")

    # print(f"\nKNN - Train Accuracy: {train_accuracy:.3f}")
    # print(f"KNN - Test Accuracy: {test_accuracy:.3f}")

    y_train_pred = best_knn.predict(X_train)
    y_test_pred = best_knn.predict(X_test)
    y_pred_proba = best_knn.predict_proba(X_test)

    # Calculate Metrics and Plot Confusion Matrix
    train_precision, train_recall, train_accuracy, train_specificity, train_f1_score, train_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_train, y_train_pred, len(np.unique(y_train)), 'train', 'KNN')
    test_precision, test_recall, test_accuracy, test_specificity, test_f1_score, test_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_test, y_test_pred, len(np.unique(y_test)), 'test', 'KNN')

    auc = fn.plot_roc_curve_with_auc(y_test, y_pred_proba, 'KNN')
    auc_ovo = fn.plot_roc_curve_with_auc_ovo(X_test, y_test, best_knn, 'KNN')
    display_metrics_console(train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, test_specificity,
                            auc, test_conf_mat)

    # print(f"\nBelow are the metrics for KNN Grid Search:\n")
    metrics.append({
        'model': 'KNN',
        'Training Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Precision': test_recall,
        'AUC': auc,
        'Training Recall': train_recall,
        'Recall': test_recall,
        'Training Specificity': train_specificity,
        'Specificity': test_specificity,
        'Training F1 Score': train_f1_score,
        'F1 Score': test_f1_score,
        'Confusion Matrix': test_conf_mat
    })

    # print(metrics)
    end_time = datetime.now()
    print("\nKNN Grid Search end-time:", end_time)
    print("\nKNN Grid Search time taken:", end_time - start_time)

    return metrics

#Function to perform Naive Bayes Classifier
def naive_bayes(X_train, X_test, y_train, y_test):
    start_time = datetime.now()
    print("\nNaive Bayes Grid Search Start-time:", start_time)
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }
    nb = GaussianNB()
    grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_nb = grid_search.best_estimator_
    train_accuracy = accuracy_score(y_train, best_nb.predict(X_train))
    test_accuracy = accuracy_score(y_test, best_nb.predict(X_test))

    # print(f"\nNaive Bayes - Train Accuracy: {train_accuracy:.3f}")
    # print(f"Naive Bayes - Test Accuracy: {test_accuracy:.3f}")
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Estimator: {best_nb}")

    y_train_pred = best_nb.predict(X_train)
    y_test_pred = best_nb.predict(X_test)
    y_pred_proba = best_nb.predict_proba(X_test)

    # Calculate Metrics and Plot Confusion Matrix
    train_precision, train_recall, train_accuracy, train_specificity, train_f1_score,train_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_train, y_train_pred, len(np.unique(y_train)), 'train', 'Naive Bayes')
    test_precision, test_recall, test_accuracy, test_specificity, test_f1_score,test_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_test, y_test_pred, len(np.unique(y_test)), 'test', 'Naive Bayes')

    auc = fn.plot_roc_curve_with_auc(y_test, y_pred_proba, 'Naive Bayes')
    auc_ovo = fn.plot_roc_curve_with_auc_ovo(X_test, y_test, best_nb, 'Naive Bayes')
    display_metrics_console(train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, test_specificity,
                            auc, test_conf_mat)

    # print(f"\nBelow are the metrics for Naive Bayes Grid Search:\n")
    metrics.append( {
        'model': 'Naive Bayes',
        'Training Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Precision': test_recall,
        'AUC': auc,
        'Training Recall': train_recall,
        'Recall': test_recall,
        'Training Specificity': train_specificity,
        'Specificity': test_specificity,
        'Training F1 Score': train_f1_score,
        'F1 Score': test_f1_score,
        'Confusion Matrix': test_conf_mat
    })

    # print(metrics)
    end_time = datetime.now()
    print("\nNaive Bayes Grid Search end-time:", end_time)
    print("\nNaive Bayes Grid Search time taken:", end_time - start_time)

    return metrics

#Function to perform Neural Networks - Multi Layered Perceptron
def neural_networks(X_train, X_test, y_train, y_test):


    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    start_time = datetime.now()
    print("\nNeural Networks Grid Search Start-time:", start_time)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (150,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'learning_rate': ['constant', 'adaptive'],
        'alpha': [0.0001, 0.001],
        'early_stopping': [True]
    }
    mlp = MLPClassifier(max_iter=500, random_state=5805)
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_mlp = grid_search.best_estimator_
    train_accuracy = accuracy_score(y_train, best_mlp.predict(X_train))
    test_accuracy = accuracy_score(y_test, best_mlp.predict(X_test))

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Estimator: {best_mlp}")

    y_train_pred = best_mlp.predict(X_train)
    y_test_pred = best_mlp.predict(X_test)
    y_pred_proba = best_mlp.predict_proba(X_test)

    # Calculate Metrics and Plot Confusion Matrix
    train_precision, train_recall, train_accuracy, train_specificity, train_f1_score, train_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_train, y_train_pred, len(np.unique(y_train)), 'train', 'Neural Networks')
    test_precision, test_recall, test_accuracy, test_specificity, test_f1_score, test_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_test, y_test_pred, len(np.unique(y_test)), 'test', 'Neural Networks')

    auc = fn.plot_roc_curve_with_auc(y_test, y_pred_proba, 'Neural Networks')
    auc_ovo = fn.plot_roc_curve_with_auc_ovo(X_test, y_test, best_mlp, 'Neural Networks')

    display_metrics_console(train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, test_specificity,
                            auc, test_conf_mat)

    # print(f"\nBelow are the metrics for Neural Networks Grid Search:\n")
    metrics.append({
        'model': 'Neural Networks - MultiLayered Perceptron',
        'Training Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Precision': test_recall,
        'AUC': auc,
        'Training Recall': train_recall,
        'Recall': test_recall,
        'Training Specificity': train_specificity,
        'Specificity': test_specificity,
        'Training F1 Score': train_f1_score,
        'F1 Score': test_f1_score,
        'Confusion Matrix': test_conf_mat
    })

    # print(metrics)
    print(f"\nNeural Networks - Train Accuracy: {train_accuracy:.3f}")
    print(f"Neural Networks - Test Accuracy: {test_accuracy:.3f}")
    end_time = datetime.now()
    print("\nNeural Networks Grid Search end-time:", end_time)
    print("\nNeural Networks Grid Search time taken:", end_time - start_time)

    return metrics


def svm_classifier(X_train, X_test, y_train, y_test):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    start_time = datetime.now()
    print("\nSVM Grid Search Start-time:", start_time)

    param_grid = {
        'kernel': ['rbf', 'linear', 'poly'],
        'degree': [2],
        'tol': [1e-3, 1e-2]
    }
    svm = GridSearchCV(SVC(probability=True, random_state=5805), param_grid=param_grid, scoring='accuracy', cv=cv, n_jobs=-1)
    svm.fit(X_train, y_train)

    best_svc = svm.best_estimator_

    # Train and Test Accuracy for MLP
    train_accuracy = accuracy_score(y_train, best_svc.predict(X_train))
    test_accuracy = accuracy_score(y_test, best_svc.predict(X_test))

    print(f"\nBest Parameters: {svm.best_params_}")
    print(f"Best Estimator: {best_svc}")

    y_train_pred = best_svc.predict(X_train)
    y_test_pred = best_svc.predict(X_test)
    y_pred_proba = best_svc.predict_proba(X_test)

    # Calculate Metrics and Plot Confusion Matrix
    train_precision, train_recall, train_accuracy, train_specificity, train_f1_score, train_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_train, y_train_pred, len(np.unique(y_train)), 'train', 'SVM')
    test_precision, test_recall, test_accuracy, test_specificity, test_f1_score, test_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_test, y_test_pred, len(np.unique(y_test)), 'test', 'SVM')

    auc = fn.plot_roc_curve_with_auc(y_test, y_pred_proba, 'SVM')
    auc_ovo = fn.plot_roc_curve_with_auc_ovo(X_test, y_test, best_svc, 'SVM')
    display_metrics_console(train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, test_specificity,
                            auc, test_conf_mat)

    print(f"\nBelow are the metrics for SVM Grid Search:\n")
    metrics.append( {
        'model': 'SVM',
        'Training Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Precision': test_recall,
        'AUC': auc,
        'Training Recall': train_recall,
        'Recall': test_recall,
        'Training Specificity': train_specificity,
        'Specificity': test_specificity,
        'Training F1 Score': train_f1_score,
        'F1 Score': test_f1_score,
        'Confusion Matrix': test_conf_mat
    })

    # print(metrics)
    # print(f"SVM - Train Accuracy: {train_accuracy:.3f}")
    # print(f"SVM - Test Accuracy: {test_accuracy:.3f}")
    end_time = datetime.now()
    print("\nSVM Grid Search end-time:", end_time)
    print("\nSVM Grid Search time taken:", end_time - start_time)

    return metrics

#Function to perform Random forest classifier
def random_forest(X_train, X_test, y_train, y_test):
    warnings.filterwarnings('ignore', category=FutureWarning)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    start_time = datetime.now()
    print("\nRandom Forest Grid Search Start-time:", start_time)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=5805)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_

    # Train and Test Accuracy
    train_accuracy = accuracy_score(y_train, best_rf.predict(X_train))
    test_accuracy = accuracy_score(y_test, best_rf.predict(X_test))

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Estimator: {best_rf}")

    # print(f"\nRandom Forest - Train Accuracy: {train_accuracy:.3f}")
    # print(f"Random Forest - Test Accuracy: {test_accuracy:.3f}")

    y_train_pred = best_rf.predict(X_train)
    y_test_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)

    # Metrics and Confusion Matrix
    train_precision, train_recall, train_accuracy, train_specificity, train_f1_score, train_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_train, y_train_pred, len(np.unique(y_train)), 'train', 'Random Forest')
    test_precision, test_recall, test_accuracy, test_specificity, test_f1_score, test_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_test, y_test_pred, len(np.unique(y_test)), 'test', 'Random Forest')

    auc = fn.plot_roc_curve_with_auc(y_test, y_pred_proba, 'Random Forest')
    auc_ovo = fn.plot_roc_curve_with_auc_ovo(X_test, y_test, best_rf, 'Random Forest')
    display_metrics_console(train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, test_specificity,
                            auc, test_conf_mat)

    metrics.append({
        'model': 'Random Forest',
        'Training Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Precision': test_recall,
        'AUC': auc,
        'Training Recall': train_recall,
        'Recall': test_recall,
        'Training Specificity': train_specificity,
        'Specificity': test_specificity,
        'Training F1 Score': train_f1_score,
        'F1 Score': test_f1_score,
        'Confusion Matrix': test_conf_mat
    })

    end_time = datetime.now()
    print("\nRandom Forest Grid Search end-time:", end_time)
    print("\nRandom Forest Grid Search time taken:", end_time - start_time)

    return metrics

#Function to perform Random forest bagging classifier
def random_forest_bagging(X_train, X_test, y_train, y_test):
    warnings.filterwarnings('ignore', category=FutureWarning)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    start_time = datetime.now()
    print("\nBagging Grid Search Start-time:", start_time)
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.75, 1.0],
        'max_features': [0.5, 0.75, 1.0],
        'estimator__max_depth': [5, 10, None]
    }
    bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=5805)
    grid_search = GridSearchCV(estimator=bagging, param_grid=param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_bagging = grid_search.best_estimator_

    # Train and Test Accuracy
    train_accuracy = accuracy_score(y_train, best_bagging.predict(X_train))
    test_accuracy = accuracy_score(y_test, best_bagging.predict(X_test))

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Estimator: {best_bagging}")

    # print(f"\nBagging - Train Accuracy: {train_accuracy:.3f}")
    # print(f"Bagging - Test Accuracy: {test_accuracy:.3f}")

    y_train_pred = best_bagging.predict(X_train)
    y_test_pred = best_bagging.predict(X_test)
    y_pred_proba = best_bagging.predict_proba(X_test)

    # Metrics and Confusion Matrix
    train_precision, train_recall, train_accuracy, train_specificity, train_f1_score, train_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_train, y_train_pred, len(np.unique(y_train)), 'train', 'Bagging')
    test_precision, test_recall, test_accuracy, test_specificity, test_f1_score, test_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_test, y_test_pred, len(np.unique(y_test)), 'test', 'Bagging')

    auc = fn.plot_roc_curve_with_auc(y_test, y_pred_proba, 'Bagging')
    auc_ovo = fn.plot_roc_curve_with_auc_ovo(X_test, y_test, best_bagging, 'Bagging')
    display_metrics_console(train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, test_specificity,
                            auc, test_conf_mat)

    metrics.append({
        'model': 'Bagging',
        'Training Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Precision': test_recall,
        'AUC': auc,
        'Training Recall': train_recall,
        'Recall': test_recall,
        'Training Specificity': train_specificity,
        'Specificity': test_specificity,
        'Training F1 Score': train_f1_score,
        'F1 Score': test_f1_score,
        'Confusion Matrix': test_conf_mat
    })

    end_time = datetime.now()
    print("\nBagging Grid Search end-time:", end_time)
    print("\nBagging Grid Search time taken:", end_time - start_time)

    return metrics

#Function to perform Random forest boosting classifier
def random_forest_boosting(X_train, X_test, y_train, y_test):
    warnings.filterwarnings('ignore', category=FutureWarning)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    start_time = datetime.now()
    print("\nBoosting Grid Search Start-time:", start_time)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    boosting = GradientBoostingClassifier(random_state=5805)
    grid_search = GridSearchCV(estimator=boosting, param_grid=param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_boosting = grid_search.best_estimator_

    # Train and Test Accuracy
    train_accuracy = accuracy_score(y_train, best_boosting.predict(X_train))
    test_accuracy = accuracy_score(y_test, best_boosting.predict(X_test))

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Estimator: {best_boosting}")

    # print(f"\nBoosting - Train Accuracy: {train_accuracy:.3f}")
    # print(f"Boosting - Test Accuracy: {test_accuracy:.3f}")

    y_train_pred = best_boosting.predict(X_train)
    y_test_pred = best_boosting.predict(X_test)
    y_pred_proba = best_boosting.predict_proba(X_test)

    # Metrics and Confusion Matrix
    train_precision, train_recall, train_accuracy, train_specificity, train_f1_score, train_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_train, y_train_pred, len(np.unique(y_train)), 'train', 'Boosting')
    test_precision, test_recall, test_accuracy, test_specificity, test_f1_score, test_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_test, y_test_pred, len(np.unique(y_test)), 'test', 'Boosting')

    auc = fn.plot_roc_curve_with_auc(y_test, y_pred_proba, 'Boosting')
    auc_ovo = fn.plot_roc_curve_with_auc_ovo(X_test, y_test, best_boosting, 'Boosting')
    display_metrics_console(train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, test_specificity,
                            auc, test_conf_mat)

    metrics.append({
        'model': 'Boosting',
        'Training Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Precision': test_recall,
        'AUC': auc,
        'Training Recall': train_recall,
        'Recall': test_recall,
        'Training Specificity': train_specificity,
        'Specificity': test_specificity,
        'Training F1 Score': train_f1_score,
        'F1 Score': test_f1_score,
        'Confusion Matrix': test_conf_mat
    })

    end_time = datetime.now()
    print("\nBoosting Grid Search end-time:", end_time)
    print("\nBoosting Grid Search time taken:", end_time - start_time)

    return metrics

#Function to perform Random forest boosting classifier
def random_forest_stacking(X_train, X_test, y_train, y_test):
    warnings.filterwarnings('ignore', category=FutureWarning)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    start_time = datetime.now()
    print("\nStacking Grid Search Start-time:", start_time)
    param_grid = {
        'final_estimator__C': [0.1, 1, 10],
    }

    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=5805)),
        ('svc', SVC(probability=True, random_state=5805)),
    ]
    stacking = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())

    grid_search = GridSearchCV(estimator=stacking, param_grid=param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_stacking = grid_search.best_estimator_

    # Train and Test Accuracy
    train_accuracy = accuracy_score(y_train, best_stacking.predict(X_train))
    test_accuracy = accuracy_score(y_test, best_stacking.predict(X_test))

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Estimator: {best_stacking}")

    # print(f"\nStacking - Train Accuracy: {train_accuracy:.3f}")
    # print(f"Stacking - Test Accuracy: {test_accuracy:.3f}")

    y_train_pred = best_stacking.predict(X_train)
    y_test_pred = best_stacking.predict(X_test)
    y_pred_proba = best_stacking.predict_proba(X_test)

    # Metrics and Confusion Matrix
    train_precision, train_recall, train_accuracy, train_specificity, train_f1_score, train_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_train, y_train_pred, len(np.unique(y_train)), 'train', 'Stacking')
    test_precision, test_recall, test_accuracy, test_specificity, test_f1_score, test_conf_mat = fn.calculate_metrics_and_plot_confusion_matrix(
        y_test, y_test_pred, len(np.unique(y_test)), 'test', 'Stacking')

    auc = fn.plot_roc_curve_with_auc(y_test, y_pred_proba, 'Stacking')
    auc_ovo = fn.plot_roc_curve_with_auc_ovo(X_test, y_test, best_stacking, 'Stacking')
    display_metrics_console(train_accuracy, test_accuracy, test_precision, test_recall, test_f1_score, test_specificity,
                            auc, test_conf_mat)

    metrics.append({
        'model': 'Stacking',
        'Training Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Precision': test_recall,
        'AUC': auc,
        'Training Recall': train_recall,
        'Recall': test_recall,
        'Training Specificity': train_specificity,
        'Specificity': test_specificity,
        'Training F1 Score': train_f1_score,
        'F1 Score': test_f1_score,
        'Confusion Matrix': test_conf_mat
    })

    end_time = datetime.now()
    print("\nStacking Grid Search end-time:", end_time)
    print("\nStacking Grid Search time taken:", end_time - start_time)

    return metrics

#Function to pre-process and balance dataset
def preprocess_and_balance(df, datetime_col, target_col, balance_method="oversample"):
    # Step 1: Convert datetime column to pandas datetime object
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Step 2: Extract features from datetime
    df['year'] = df[datetime_col].dt.year
    df['month'] = df[datetime_col].dt.month
    df['day'] = df[datetime_col].dt.day
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    df['is_weekend'] = df[datetime_col].dt.dayofweek >= 5

    # Step 3: Drop the original datetime column
    df = df.drop(columns=[datetime_col])

    # Step 4: Normalize numerical datetime features
    scaler = MinMaxScaler()
    df[['month', 'day', 'hour', 'day_of_week']] = scaler.fit_transform(df[['month', 'day', 'hour', 'day_of_week']])

    # Step 5: Balance the dataset
    balanced_df = pd.DataFrame()
    classes = df[target_col].unique()

    if balance_method == "undersample":
        min_class_size = df[target_col].value_counts().min()
        for cls in classes:
            class_subset = df[df[target_col] == cls]
            balanced_subset = resample(class_subset, replace=False, n_samples=min_class_size, random_state=5805)
            balanced_df = pd.concat([balanced_df, balanced_subset])
    elif balance_method == "oversample":
        max_class_size = df[target_col].value_counts().max()
        for cls in classes:
            class_subset = df[df[target_col] == cls]
            balanced_subset = resample(class_subset, replace=True, n_samples=max_class_size, random_state=5805)
            balanced_df = pd.concat([balanced_df, balanced_subset])
    else:
        raise ValueError("Invalid balance_method. Use 'undersample' or 'oversample'.")

    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=5805).reset_index(drop=True)

    return balanced_df