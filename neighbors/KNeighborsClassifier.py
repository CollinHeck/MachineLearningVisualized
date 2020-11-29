import numpy as np


class KNeighborsClassifier():

    # TODO: Add classification info
    def __init__(self, X=None, k=3):
        self.X = X
        self.k = k

    def fit(self, X):
        self.X = X

    def setK(self, k):
        self.k = k

    def predict(self, y):
        # TODO: Fix the outputs

        distances = []
        rows = []
        counter = 0
        for row in self.X:
            distances.append(self.euclideanDistance(row, y))
            rows.append(counter)
            counter += 1

        out = zip(distances, rows)
        print(tuple(out))

    @staticmethod
    def euclideanDistance(arr1, arr2):
        assert arr1.shape == arr2.shape, "Arrays need to be equal shapes for Euclidean Distance."
        return np.sqrt(np.sum(np.square(arr1 - arr2)))

    # TODO: Add visuals
