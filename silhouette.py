from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm



def sil_func(X, name, number):
    if (name == "gmm"):
        gmm = GaussianMixture(n_components=number, n_init=30, verbose_interval=10, random_state=10)
        gmm.fit(X)
        labels = gmm.predict(X)
    elif (name == "k-means"):
        model = KMeans(n_clusters=number)
        model.fit(X)
        labels = model.predict(X)

    # Create a subplot with 1 row and 2 columns
    fig, (ax1) = plt.subplots(1, 1)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (8+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (number + 1) * 10])

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, labels)
    print("silhouette score: ", end=" ")
    print(sum(sample_silhouette_values) / len(sample_silhouette_values))

    y_lower = 10
    for i in range(number):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / number)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    if (name == "gmm"):
        ax1.set_title("The silhouette plot for the various clusters - gmm.")
    elif (name == "k-means"):
        ax1.set_title("The silhouette plot for the various clusters - k-means.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    plt.show()


def main(dataset_x, name, number):
    sil_func(dataset_x[: 5000], name, number)
