from typing import List
import numpy as np


class Classifier:
    def __init__(self, features):
        self.features = features

    def classify(self, example) -> bool:
        for feature, value in self.features.items():
            if example[feature] != value:
                return False
        return True

    def get_accuracy(self, validation_set, validation_labels) -> (float):
        """

        :param validation_set:
        :param validation_labels:
        :return: accuracy [0,1]
        """
        correct = 0
        for i in range(len(validation_set)):
            if self.classify(validation_set[i]) == validation_labels[i]:
                correct += 1
        return correct / len(validation_set)

    def get_matrix(self, validation_set, validation_labels):
        #   + -
        # + 00 01
        # - 10 11
        matrix = np.matrix([[0, 0], [0, 0]])
        for i in range(len(validation_set)):
            if self.classify(validation_set[i]) == validation_labels[i]:
                if validation_labels[i] == True:  # true positive
                    matrix[0, 0] += 1
                else:  # true negative
                    matrix[1, 1] += 1
            else:
                if validation_labels[i] == True:  # false negative
                    matrix[1, 0] += 1
                else:  # false positive
                    matrix[0, 1] += 1

        return matrix


class Classifier_factory:
    def __init__(self, data: List[List[int]], labels: List[bool], algorithm):
        self.data = data
        self.labels = labels
        self.algorithm = algorithm

    def train(self, data: List[List[int]], labels: List[bool]):
        """

        :param data:
        :param labels:
        :return: classifier, average accuracy of all folds
        """

        alg = self.algorithm

        examples_num = len(data)
        features_num = len(data[0])
        fold_size = int(np.ceil(examples_num / 4))
        folds = []
        folds.append(data[0:fold_size - 1])
        folds.append(data[fold_size:fold_size * 2 - 1])
        folds.append(data[fold_size * 2:fold_size * 3 - 1])
        folds.append(data[fold_size * 3:])

        labels_ = []
        labels_.append(labels[0:fold_size - 1])
        labels_.append(labels[fold_size:fold_size * 2 - 1])
        labels_.append(labels[fold_size * 2:fold_size * 3 - 1])
        labels_.append(labels[fold_size * 3:])

        features_lists = []
        best_accuracy = -1
        best_classifier = Classifier([])
        accuracy_avg = 0
        matrix_avg = np.matrix([[0, 0], [0, 0]])
        for i in range(4):
            set_ = list(set(range(4)) - set([i]))
            train_set = folds[set_[0]] + folds[set_[1]] + folds[set_[2]]
            features_lists.append(alg.run_algorithm(range(features_num), train_set))
            classifier = Classifier(features_lists[i])
            accuracy = classifier.get_accuracy(folds[i], labels_[i])
            accuracy_avg += accuracy
            matrix = classifier.get_matrix(folds[i], labels_[i])
            matrix_avg = matrix_avg + matrix
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classifier = classifier
        accuracy_avg = accuracy_avg / 4
        matrix_avg = 0.25 * matrix_avg
        return best_classifier, accuracy_avg, matrix_avg

    def get_accuracy(self, validation_set, validation_labels, features_list) -> (float, Classifier):
        """

        :param validation_set:
        :param validation_labels:
        :param features_list:
        :return: accuracy [0,1]
        """

        classifier = Classifier(features_list)
        correct = 0
        for i in range(len(validation_set)):
            if classifier.classify(validation_set[i]) == validation_labels[i]:
                correct += 1
        return (correct / len(validation_set), classifier)
