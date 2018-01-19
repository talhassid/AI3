import csv
from typing import List

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def load_data(path: str) -> List[int]:
    """
    Load the dataset and return as a list

    :param path:  path to the flare.csv
    :return: the features per example and it's label
    """
    with open(path, 'r') as f:
        data = list(csv.reader(f, delimiter=","))
        for e in range(1, len(data[1:]) + 1):
            for f in range(len(data[e]) - 1):
                data[e][f] = int(data[e][f])
            data[e][-1] = (data[e][-1] == "True")
    return data[1:]


if __name__ == '__main__':
    path = "./flare.csv"
    data = load_data(path)
    data_ = [row[:-1] for row in data[1:]]
    labels_ = [row[-1] for row in data[1:]]

    x_train, x_test, y_train, y_test = train_test_split(data_, labels_, test_size=0.25)
    clf_overfitting = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=1, min_samples_split=2)
    clf_overfitting.fit(X=x_train, y=y_train)
    print(clf_overfitting.score(X=x_train, y=y_train))

    clf_underfitting = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=20, min_samples_split=20)
    clf_underfitting.fit(X=x_train, y=y_train)
    print(clf_underfitting.score(X=x_train, y=y_train))
