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
        #print(type(x))
        output = self.params[0]
        for m in x:
            #print(m)
            multiset_out = 0
            for v in m:
                #print(v)
                multiset_out += v[0] * v[1]
            output += multiset_out
        return self.sigmoid(output)
    
    def sigmoid(self,x):
        return 1/ (1+np.exp(-x)) 
    
class Identity():
    def forward(x=None):
        return 1
    def __str__(self):
        return "Identity"

