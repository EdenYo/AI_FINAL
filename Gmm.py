from category_encoders import TargetEncoder
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import operator
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
import silhouette



def read_data(csv_file_path):
    dataset = pd.read_csv(csv_file_path)
    dataset.columns = ['edible', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                       'gill-spacing',
                       'gill-size', 'gill-color', 'stalk-shape', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                       'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                       'ring-type', 'spore-print-color', 'population', 'habitat']

    dataset_y = dataset.pop('odor')
    dataset_x = dataset

    return dataset_x, dataset_y

def preprocessing(dataset_x, dataset_y):
    label_encoder = LabelEncoder()
    dataset_y = label_encoder.fit_transform(dataset_y)
    target_encoder = TargetEncoder()
    dataset_x = target_encoder.fit_transform(dataset_x, dataset_y)

    return dataset_x, dataset_y

def write_predicts(X):
    gmm = GaussianMixture(n_components=10, n_init=30, verbose_interval=10, random_state=10)
    gmm.fit(X)
    labels = gmm.predict(X)
    labels = labels.ravel()+1
    pd.DataFrame(labels).to_csv("clustering_gmm.csv", header=None, index=None)
    return labels

def plot_pca(dataset_x, labels):
    pca = PCA(n_components=3)
    dataset_x_reduced = pca.fit_transform(dataset_x)
    fig = plt.figure(figsize=(4, 4))

    ax = fig.add_subplot(projection='3d')
    colors = ['b', 'g', 'r', 'c', 'm', 'yellow', 'orange', 'fuchsia', 'silver', 'pink']
    for i in range(6498):
        ax.scatter(dataset_x_reduced[i, 0], dataset_x_reduced[i, 1], dataset_x_reduced[i, 2], color=colors[labels[i]-1])
    ax.set_title("10 clusters with GMM")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.show()




def cal_scores(X, cluster_sizes):
    bic_scores = []
    aic_scores = []
    seed = 10
    for n_clusters in cluster_sizes:
        gmm = GaussianMixture(n_components=n_clusters, n_init=30, verbose_interval=10, random_state=seed)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))
    return bic_scores, aic_scores

def main(csv_file_path,is_PCA):
    dataset_x, dataset_y = read_data(csv_file_path)
    dataset_y_before_encoder = dataset_y
    dataset_x, dataset_y = preprocessing(dataset_x, dataset_y)
    if is_PCA == False:
        cluster_sizes = np.arange(1, 25)
        bic_scores, aic_scores = cal_scores(dataset_x, cluster_sizes)
        plt.plot(cluster_sizes, bic_scores, '-o', label='BIC', c='r')
        plt.plot(cluster_sizes, aic_scores, '-o', label='AIC', c='b')
        plt.title("BIC and AIC")
        plt.xlabel("number of clusters")
        plt.ylabel("BIC and AIC")
        plt.show()
    if is_PCA:
        transformer = FastICA(n_components=4, random_state=0)
        dataset_x = transformer.fit_transform(dataset_x)
    labels = write_predicts(dataset_x)
    # display silhouette score
    silhouette.main(dataset_x, "gmm", 10)

    # now we want to classify the sample to the correct odor
    # we will take the most common odor in the cluster

    cluster_count = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    for i in range(len(labels)):
        dict = cluster_count[labels[i] - 1]
        dict[dataset_y_before_encoder[i]] = dict.get(dataset_y_before_encoder[i], 0) + 1
    pred = ['a' for x in range(len(labels))]
    for i in range(len(labels)):
        pred[i] = max(cluster_count[labels[i] - 1].items(), key=operator.itemgetter(1))[0]

    # confusion matrix
    data = {'y_Actual': dataset_y_before_encoder, 'y_Predicted': pred}
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True, cmap="Purples")
    plt.title("Gmm - confusion matrix")
    plt.show()
    print("GMM Classifier report: \n\n", classification_report(dataset_y_before_encoder, pred, zero_division=1))
    plot_pca(dataset_x, labels)

