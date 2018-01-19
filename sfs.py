import csv
from typing import List

def sfs(x, y, k, clf, score):
    """
    :param x: feature set to be trained using clf. list of lists.
    :param y: labels corresponding to x. list.
    :param k: number of features to select. int
    :param clf: classifier to be trained on the feature subset.
    :param score: utility function for the algorithm, that receives clf, feature subset and labeles, returns a score.
    :return: list of chosen feature indexes
    """
    features = range(len(x[0]))
    selected_features = []
    features_not = features
    for i in range(k):
        feature = select_feature(score=score, clf=clf, data=x, features=features_not,
                                 selected_features=selected_features, labels=y)
        selected_features.append(feature)
        features_not = list(set(features_not) - set([feature]))
    return selected_features

def select_feature(score, clf, data, features, selected_features, labels):
    max_score = 0
    max_f = features[0]
    for f in features:
        features_subset = [[row[ind] for ind in [f] + selected_features] for row in data]
        f_score = score(clf, features_subset, labels)
        if f_score > max_score:
            max_score = f_score
            max_f = f
    return max_f

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

def score(clf, features, labels) -> float:
    clf.fit(features, labels)
    return clf.score(features, labels)

