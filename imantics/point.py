import numpy as np


class Point(np.ndarray):

    def __abs__(self):
        return np.linalg.norm(self)

    def dist(self,other):
        return np.linalg.norm(self-other)

    def dot(self, other):
        return np.dot(self, other)

