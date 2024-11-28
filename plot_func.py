import matplotlib.pyplot as plt
import numpy as np
import func as fn

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

    # Plot the class distribution
    class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Dataset Class distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.xticks([0,1,2], labels=['delayed','on-time','early'])
    plt.show()
