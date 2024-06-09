import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kmeans(X, n_clusters, max_iters=100):
    centroids = X[np.random.choice(range(len(X)), size=n_clusters, replace=False)]
    for _ in range(max_iters):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels

def gaussian_mixture(X, n_clusters, max_iters=100):
    # Initialize parameters randomly
    np.random.seed(0)
    pi = np.random.dirichlet(np.ones(n_clusters))
    mu = X[np.random.choice(range(len(X)), size=n_clusters, replace=False)]
    sigma = np.array([np.eye(len(X[0])) for _ in range(n_clusters)])

    # Expectation-Maximization algorithm
    for _ in range(max_iters):
        # Expectation step
        likelihoods = np.array([pi[k] * multivariate_normal.pdf(X, mu[k], sigma[k]) for k in range(n_clusters)])
        responsibilities = likelihoods / likelihoods.sum(axis=0)

        # Maximization step
        Nk = responsibilities.sum(axis=1)
        pi = Nk / len(X)
        mu = np.dot(responsibilities, X) / Nk[:, None]
        sigma = np.array([np.dot((X - mu[k]).T, (responsibilities[k][:, None] * (X - mu[k]))) / Nk[k] for k in range(n_clusters)])

    return np.argmax(responsibilities, axis=0)

def main():
    st.write("22AIA-MACHINE MASTERS")
    st.title("Iris Clustering Visualization")

    # Load the Iris dataset
    from sklearn.datasets import load_iris
    dataset = load_iris()
    X = dataset.data
    y = dataset.target

    # Real Plot
    plt.figure(figsize=(14, 7))
    colormap = np.array(['red', 'lime', 'black'])
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 2], X[:, 3], c=colormap[y], s=40)
    plt.title('Real')

    # KMeans Plot
    plt.subplot(1, 3, 2)
    kmeans_labels = kmeans(X[:, 2:], n_clusters=3)
    plt.scatter(X[:, 2], X[:, 3], c=colormap[kmeans_labels], s=40)
    plt.title('KMeans')

    # Gaussian Mixture Model Plot
    plt.subplot(1, 3, 3)
    gmm_labels = gaussian_mixture(X[:, 2:], n_clusters=3)
    plt.scatter(X[:, 2], X[:, 3], c=colormap[gmm_labels], s=40)
    plt.title('GMM Classification')

    # Display the plots
    st.pyplot()

if __name__ == "__main__":
    main()
