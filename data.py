import csv
from typing import List


def load_data(path: str) -> (List[int], List[int]):
    """
    Load the dataset and return as a list

    :param path:  path to the flare.csv
    :return: the features List and labels List
    """
    with open(path, 'r') as f:
        data = list(csv.reader(f, delimiter=","))
        for e in range(1, len(data[1:]) + 1):
            for f in range(len(data[e]) - 1):
                data[e][f] = int(data[e][f])
            data[e][-1] = (data[e][-1] == "True")

    data_ = [row[:-1] for row in data[1:]]
    labels_ = [row[-1] for row in data[1:]]
    return data_, labels_


if __name__ == '__main__':
    path = "./flare.csv"
    data, labels = load_data(path)
