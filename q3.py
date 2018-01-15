import csv
from typing import List

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
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
    clf = DecisionTreeClassifier(criterion="entropy")

    k_fold = 4
    predict_over_4_folds = cross_val_score(estimator=clf, X=data_, y=labels_, cv=k_fold)

    clf0 = DecisionTreeClassifier(criterion="entropy")
    clf0.fit(X=data_, y=labels_)
    prediction_list = list(clf0.predict(X=data_))
    # print("Confusion Matrix")
    # print(confusion_matrix(y_true=data_, y_pred=prediction_list, labels=labels_))

    print("-"*30,"\nDefault values:")
    clf1 = DecisionTreeClassifier(criterion="entropy")
    clf1.fit(X=data_[:318], y=labels_[:318])
    print("Train accuracy" , clf1.score(X=data_[:318], y=labels_[:318]))
    prediction_list = clf1.predict(X=data_[319:])
    print("Test accuracy" , clf1.score(X=data_[319:],y=labels_[319:]))


    print("-"*30,"\nUnderFitting:")
    clf2 = DecisionTreeClassifier(criterion="entropy",min_samples_leaf=20,min_samples_split=20)
    clf2.fit(X=data_[:318], y=labels_[:318])
    print("Train accuracy with prunning" , clf2.score(X=data_[:318], y=labels_[:318]))
    prediction_list = clf2.predict(X=data_[319:])
    print("Test accuracy with prunning" , clf2.score(X=data_[319:],y=labels_[319:]))

    print("-"*30,"\nOverFitting:")
    clf3 = DecisionTreeClassifier(criterion="entropy",min_samples_leaf=1,min_samples_split=2)
    clf3.fit(X=data_[:318], y=labels_[:318])
    print("Train accuracy with prunning" , clf3.score(X=data_[:318], y=labels_[:318]))
    prediction_list = clf3.predict(X=data_[319:])
    print("Test accuracy with prunning" , clf3.score(X=data_[319:],y=labels_[319:]))