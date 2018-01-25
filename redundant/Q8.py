from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data import load_data

path = "./flare.csv"
data, labels = load_data(path)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(X=x_train, y=y_train)

print("Decision Tree")
print("train {} test {}".format(dt.score(X=x_train, y=y_train), dt.score(X=x_test, y=y_test)))
dt_p = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=20)
dt_p.fit(X=x_train, y=y_train)

print("Decision Tree with Pruning")
print("train {} test {}".format(dt_p.score(X=x_train, y=y_train), dt_p.score(X=x_test, y=y_test)))
