import numpy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from data import load_data

if __name__ == '__main__':
    path = "./flare.csv"
    data, labels = load_data(path)

    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X=data, y=labels)
    k_fold = 4
    score_over_4_folds = cross_val_score(estimator=clf, X=data, y=labels, cv=k_fold)
    print(numpy.average(score_over_4_folds))

    y_pred = cross_val_predict(estimator=clf, X=data, y=labels)
    print(confusion_matrix(y_true=labels, y_pred=y_pred))
