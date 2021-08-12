import os
from sklearn.metrics import classification_report
import ML
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

if __name__ == '__main__':
    csv_file_path = os.path.join(os.path.dirname(__file__), 'Data', 'mushrooms_data.txt')
    x_train, x_test, y_train, y_test, dataset_x, dataset_y = ML.read_data(csv_file_path, False)
    clf = ML.fit(x_train, y_train, "second_assistant")
    y_pred = clf.predict(x_test)
    pd.DataFrame(y_pred).to_csv("SVM_classification.csv", header=None, index=None)
    print("SVM Classifier report: \n\n", classification_report(y_test, y_pred, zero_division=1))
    print("Test Accuracy: {}%".format(round(clf.score(x_test, y_test) * 100, 2)))
    data = {'y_Actual': y_test,
            'y_Predicted': y_pred
            }

    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    sn.heatmap(confusion_matrix, annot=True, cmap="YlGnBu")
    plt.title("SVM - confusion matrix")
    plt.show()