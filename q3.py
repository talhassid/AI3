from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data import load_data

if __name__ == '__main__':
    path = "./flare.csv"
    data, labels = load_data(path)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
    clf_overfitting = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=1, min_samples_split=2)
    clf_overfitting.fit(X=x_train, y=y_train)
    print(clf_overfitting.score(X=x_train, y=y_train))

    clf_underfitting = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=20, min_samples_split=20)
    clf_underfitting.fit(X=x_train, y=y_train)
    print(clf_underfitting.score(X=x_train, y=y_train))
