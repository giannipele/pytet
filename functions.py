import autograd.numpy as np
from fractions import Fraction

class Logistic():
    def __init__(self, params):
        self.params = np.array(params)
    def __str__(self):
        return "Logistic, {}".format(self.params)

    def forward(self, x=None):
        if x.all() == None:
            return 0
        output = self.params[0]
        i = 1
        for m in x:
            multiset_out = 0
            for v in m:
                multiset_out += self.params[i] * v[0] * v[1]
            i += 1
            output += multiset_out
        return self.sigmoid(output)
    
    def sigmoid(self,x):
        return 1/ (1+np.exp(-x)) 
    
class Identity():
    def __init__(self):
        self.params = []

    def forward(self, x=None):
        return 1

    def __str__(self):
        return "Identity"

