import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import apriori, association_rules

def kmeans_clustering(data):

    data_numeric = data.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)
    wcss = []
    silhouette_scores = []
    k_range = range(2, 21)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=5805, max_iter=300, tol=1e-4)
        labels = kmeans.fit_predict(data_scaled)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data_scaled, labels))
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(k_range, wcss, marker='o')
    ax[0].set_title('Elbow Method')
    ax[0].set_xlabel('Number of Clusters')
    ax[0].set_ylabel('WCSS')

    ax[1].plot(k_range, silhouette_scores, marker='o')
    ax[1].set_title('Silhouette Analysis')
    ax[1].set_xlabel('Number of Clusters')
    ax[1].set_ylabel('Silhouette Score')
    plt.show()
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nK-Means\nOptimal K based on Silhouette Score: {optimal_k}")
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=5805, max_iter=300)
    clusters = kmeans.fit_predict(data_scaled)

    return clusters


def dbscan_clustering(data, min_samples=5):

    data_numeric = data.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(data_scaled)
    distances, indices = neighbors_fit.kneighbors(data_scaled)
    distances = np.sort(distances[:, -1])
    plt.plot(distances)
    plt.title('K-Distance Graph for DBSCAN')
    plt.xlabel('Data points sorted by distance')
    plt.ylabel('Eps distance')
    plt.grid(True)
    plt.show()
    eps_value = 0.5
    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples, n_jobs=-1)
    clusters = dbscan.fit_predict(data_scaled)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"\nDB-SCAN\nEstimated number of clusters: {n_clusters}")
    return clusters

def apriori_analysis(transactions_df, min_support=0.05, min_confidence=0.1):
    transactions_df = transactions_df.select_dtypes(include=[object])
    encoded=pd.get_dummies(transactions_df, drop_first=True)
    transactions_df=encoded

    frequent_itemsets = apriori(transactions_df, min_support=min_support, use_colnames=True, low_memory=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=2)
    strong_rules = rules[rules['lift'] > 1.2].sort_values(by='lift', ascending=False)
    print(f"\nAssociation rules:\n{rules}")
    print(f"\nFrequent Item sets:\n{frequent_itemsets}")

    return strong_rules
