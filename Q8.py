import csv
from typing import List

import numpy
from sklearn.model_selection import train_test_split, cross_val_score
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


path = "./flare.csv"
data = load_data(path)
data_ = [row[:-1] for row in data[1:]]
labels_ = [row[-1] for row in data[1:]]


x_train, x_test, y_train, y_test = train_test_split(data_,labels_,test_size=0.25)
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(X=x_train, y=y_train)

print("Decision Tree")
print("{} {}".format(dt.score(X=x_train,y=y_train),dt.score(X=x_test,y=y_test)))
dt_p = DecisionTreeClassifier(criterion="entropy",min_samples_leaf=20)
dt_p.fit(X=x_train, y=y_train)

print("Decision Tree with Pruning")
print("{} {}".format(dt_p.score(X=x_train,y=y_train),dt_p.score(X=x_test,y=y_test)))
