import autograd.numpy as np
from fractions import Fraction
from abc import ABC, abstractmethod
import autograd.numpy as np

class Logistic():
    def __init__(self, params):
        self.params = np.array(params)

    def __str__(self):
        return "Logistic, {}".format(self.params)

    def forward(self, params, x):
        output = params[0] + np.sum([np.sum([params[i+1] * x[i][v][0] * x[i][v][1] for v in range(len(x[i]))]) for i in range(len(x))])
        return self.sigmoid(output)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    '''def forward(self, params, x):
       output = params[0]
       i = 1
       for m in x:
           multiset_out = 0
           for v in m:
               multiset_out += params[i] * v[0] * v[1]
           i += 1
           output += multiset_out
       return self.sigmoid(output)'''


class Identity():
    def __init__(self):
        self.params = []

    def forward(self, x=None):
        return 1

    def __str__(self):
        return "Identity"


class LossFunction():
    @abstractmethod
    def loss(self, y_hat, y):
        pass


class MeanSquaredError(LossFunction):
    def loss(self, y_hat, y):
        assert len(y_hat) == len(y), "Label vectors differ in size."
        num = 0
        for y_h, y_t in zip(y_hat, y):
            num = num + (y_h - y_t)**2
        return num/(2*len(y_hat))


class CrossEntropy(LossFunction):
    def loss(self, y_hat, y):
        assert len(y_hat) == len(y), "Label vectors differ in size."
        eps = np.finfo(float).eps
        num = 0
        for y_h, y_t in zip(y_hat, y):
            num = num - ((y_t*np.log(y_h + eps) + (1-y_t)*np.log(1-y_h + eps)))
        return num/len(y_hat)
