import numpy as np


class id3:
    def __init__(self, classes):
        """
        Create Id3 algorithem instance
        """
        self.classes = classes

    def entropy(self, E):
        entropy = 0
        if E:
            for c in self.classes:
                Ec = [element for element in E if element == c]
                Pc = len(Ec) / len(E)
                if Pc > 0:
                    entropy = entropy - Pc * np.log2(Pc)
            return entropy
        else:
            return 0

    def information_gain(self, f, E):
        current_entropy = self.entropy(E)
        children_entropy = 0

        Ve = [e[f] for e in E]
        set_V = set(Ve)
        for v in set_V:
            Ev = [e for e in E if e[f] == v]
            children_entropy = children_entropy + (len(Ev) / len(E)) * self.entropy(Ev)
        return current_entropy - children_entropy

    def select_feature(self, features, E):
        max_gain = 0
        max_f = features[0]
        for f in features:
            f_gain = self.information_gain(f, E)
            if f_gain > max_gain:
                max_gain = f_gain
                max_f = f
        return max_f

    def run_algorithm(self, features, E):
        features_ = features
        features_list = {}
        for i in range(len(features)):
            selected_feature = self.select_feature(features_, E)  # chose the feature with the highest IG
            features_ = list(
                set(features_) - set([selected_feature]))  # remove the selected_feature from the features group
            features_list.update({selected_feature: 0})
            Ve = [e[selected_feature] for e in E]  # devide the examples by the selected_feature value
            set_V = set(Ve)
            entropy = 1
            Ev_min = []
            for v in set_V:  # select the example group with the minimum entropy
                Ev = [e for e in E if e[selected_feature] == v]
                entropy_Ev = self.entropy(Ev)
                if entropy_Ev <= entropy:
                    entropy = entropy_Ev
                    Ev_min = Ev
                    features_list[selected_feature] = Ev_min[0][selected_feature]

            if entropy == 0 or entropy == 1:
                break
        return features_list
