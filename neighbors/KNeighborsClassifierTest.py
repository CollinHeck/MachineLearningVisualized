# Temporarily test KNN class during development.


import numpy as np

from KNeighborsClassifier import *

knn = KNeighborsClassifier()

a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 6, 7, 8, 9])

e = knn.euclideanDistance(a,b)
print(e)


fitA = np.array([[4, 3, 3, 4, 9],
                 [1, 2, 3, 4, 5],
                 [5, 6, 7, 8, 9],
                 [0, 0, 3, 4, 6],
                 [11, 22, 34, 45, 63],
                 [5, 4, 3, 2, 100],
                 [1, 2, 3, 4, 1540000]])

knn.fit(fitA)
knn.setK(3)

knn.predict(a)

print('Next')

knn.predict(b)

print('Next')

print(fitA.shape)
print(np.arange(0,fitA.shape[0]))