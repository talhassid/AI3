from typing import List
import csv

import numpy
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from classifier import Classifier_factory
from id3 import id3


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

    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X=data_, y=labels_)
    k_fold = 4
    score_over_4_folds = cross_val_score(estimator=clf, X=data_, y=labels_, cv=k_fold)
    print(numpy.average(score_over_4_folds))

    y_pred = cross_val_predict(estimator=clf, X=data_,y=labels_)
    print(confusion_matrix(y_true=labels_, y_pred=y_pred))

    # alg = id3([True,False])
    # c_f = Classifier_factory(data_, labels_, alg)
    # classifier, acc, mat = c_f.train(data_, labels_)
    # print(acc)
    # print(mat)
