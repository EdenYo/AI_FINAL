
import ML
import os
import pandas as pd
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# treat the missing values as a separate category by itself.
def first_approach():
    csv_file_path = os.path.join(os.path.dirname(__file__), 'Data', 'mushrooms_data_missing.txt')
    df = pd.read_csv(csv_file_path)
    df.columns = ['edible', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                       'gill-spacing',
                       'gill-size', 'gill-color', 'stalk-shape', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                       'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                       'ring-type', 'spore-print-color', 'population', 'habitat']
    # print the number of missing values for each feature
    for x in range(22):
        print('Missing values for the ' + x.__str__() + ' feature' , (df.iloc[:, x] == '-').sum())
    x_train, x_test, y_train, y_test, dataset_x, dataset_y = ML.read_data(csv_file_path, False)
    clf = ML.fit(x_train, y_train, "missing_data")
    y_pred = clf.predict(x_test)
    print("random forest Classifier report: \n\n", classification_report(y_test, y_pred, zero_division=1))
    print("Test Accuracy: {}%".format(round(clf.score(x_test, y_test) * 100, 2)))

def knn():
    # load dataset
    csv_file_path = os.path.join(os.path.dirname(__file__), 'Data', 'mushrooms_data_missing.txt')
    dataframe = read_csv(csv_file_path, header=None)
    dataframe.columns = ['edible', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                         'gill-spacing',
                         'gill-size', 'gill-color', 'stalk-shape', 'stalk-surface-above-ring',
                         'stalk-surface-below-ring',
                         'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                         'ring-type', 'spore-print-color', 'population', 'habitat']
    print("veil-type unique values: ")
    print(dataframe['veil-type'].unique())
    dataframe.pop('veil-type')

    features = dataframe.iloc[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    features = pd.get_dummies(features).astype(float)  # one hot encoding
    target = dataframe.iloc[:, 5]
    dataset_x = features.values
    dataset_y = target.values
    # split into input and output elements
    data = dataset_x
    ix = [i for i in range(data.shape[1]) if i != 21]
    X, y = data[:, ix], data[:, 21]
    # define imputer
    imputer = KNNImputer()
    # fit on the dataset
    imputer.fit(X)
    # transform the dataset
    dataset_x = imputer.transform(X)

    x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.2)
    clf = RandomForestClassifier(min_samples_leaf=10, min_samples_split=30, n_estimators=200, random_state=42)
    clf.fit(x_train, y_train)
    print("random forest, train score:" + str(clf.score(x_train, y_train)))
    y_pred = clf.predict(x_test)

    print("Random forest Classifier report: \n\n", classification_report(y_test, y_pred, zero_division=1))
    print("Test Accuracy: {}%".format(round(clf.score(x_test, y_test) * 100, 2)))

if __name__ == '__main__':
    # second part of the assignment - missing data
    first_approach()
    print("***********************************************")
    knn()






