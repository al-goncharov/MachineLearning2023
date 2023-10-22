import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator

def H(y, R):
    y_1 = y
    y_0 = abs(y - 1)
    y_1 = np.multiply(R, y_1)
    y_0 = np.multiply(R, y_0)
    p0 = np.linalg.norm(y_0, ord=1, axis=1) / np.linalg.norm(R, ord=1, axis=1)
    p1 = np.linalg.norm(y_1, ord=1, axis=1) / np.linalg.norm(R, ord=1, axis=1)
    h = 1 - np.square(p0) - np.square(p1)
    return h

def gini(R_l, R_r, y):
    R = R_l + R_r
    gini_l = np.linalg.norm(R_l, ord=1, axis=1) / np.linalg.norm(R, ord=1, axis=1)
    gini_r = np.linalg.norm(R_r, ord=1, axis=1) / np.linalg.norm(R, ord=1, axis=1)
    h_l = H(y,R_l)
    h_r = H(y,R_r)
    Q = -np.multiply(gini_l, h_l) -np.multiply(gini_r, h_r)
    return Q

def find_best_split(feature_vector, target_vector):

    index = np.argsort(feature_vector)
    feature_vector = np.array(feature_vector)[index]
    target_vector = np.array(target_vector)[index]
    thresholds = (feature_vector[1:] + feature_vector[:-1]) / 2
    thresholds = np.unique(thresholds)

    R_l = (feature_vector) < thresholds.reshape(-1,1)
    R_r = ~R_l
    index = R_l.any(axis=1) & R_r.any(axis=1)
    index = np.squeeze(index.reshape(1, -1))
    thresholds = thresholds[index]
    R_l = np.matrix(feature_vector) < thresholds.reshape(-1,1)
    R_r = ~R_l
    
    m = R_l.shape[0]
    y = np.zeros((m, 1)) + target_vector
    
    if len(R_l) == 0 or len(R_r) == 0:
            return [], [], -np.inf, -np.inf
    
    ginis = gini(R_l, R_r, y)
    best_gini = np.argmax(ginis)
    threshold_best = thresholds[best_gini]
    gini_best = ginis[best_gini]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self.feature_types = feature_types

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]): #2
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]): #3
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
                
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                        ratio[key] = current_count / current_click
                    else:
                        current_click = 0
                        ratio[key] = 0
                
                sorted_categories = sorted(ratio.keys(),
                                           key=lambda k: ratio[k]) 
                
                categories_map = dict(zip(sorted_categories,
                                          range(len(sorted_categories))))
                
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError
            
            if len(feature_vector) == 3:
                continue
                
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list( map(lambda x: x[0], 
                                filter(lambda x: x[1] < threshold,categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
            
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
            
        else:
            raise ValueError
            
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"]) #1

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        else:
            feature_split = node["feature_split"]
            
            if self.feature_types[feature_split] == "real":
                threshold = node["threshold"]
                if x[feature_split] < threshold:
                    return self._predict_node(x, node['left_child'])
                else:
                    return self._predict_node(x, node['right_child'])
                
                
            elif self.feature_types[feature_split] == "categorical":
                threshold = node["categories_split"]
                if x[feature_split] in threshold:
                    return self._predict_node(x, node['left_child'])
                else:
                    return self._predict_node(x, node['right_child'])
                    
            else:
                raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
