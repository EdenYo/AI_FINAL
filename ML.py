from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.decomposition import PCA, FastICA


def read_data(csv_file_path, is_PCA):
    dataset = pd.read_csv(csv_file_path)
    dataset.columns = ['edible', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                       'gill-spacing',
                       'gill-size', 'gill-color', 'stalk-shape', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                       'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                       'ring-type', 'spore-print-color', 'population', 'habitat']
    features = dataset.iloc[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
    features = pd.get_dummies(features).astype(float)  # one hot encoding
    target = dataset.iloc[:, 5]
    dataset_x = features.values
    dataset_y = target.values
    if is_PCA:
        pca = PCA(n_components=10)
        dataset_x = pca.fit_transform(dataset_x)
    x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.2)

    return x_train, x_test, y_train, y_test, dataset_x, dataset_y


def fit(x_train, y_train, mission):
    if mission == "missing_data":
        # random forest
        rf = RandomForestClassifier(min_samples_leaf=1, min_samples_split=2, n_estimators=200, random_state=42)
        rf.fit(x_train, y_train)
        return rf
    elif mission == "second_assistant" :
        # svm
        clf_SVM = svm.SVC(kernel='sigmoid', gamma='auto')
        # Train the model using the training sets
        clf_SVM.fit(x_train, y_train)
        return clf_SVM