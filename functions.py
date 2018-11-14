import autograd.numpy as np

class Logistic():
    def __init__(self, params):
        self.params = np.array(params)
    def __str__(self):
        return "Logistic, {}".format(self.params)

    def forward(self, x):
        print(type(x))
        output = self.params[0]
        for m in x:
            print(m)
            multiset_out = 0
            for v in m:
                print(v)
                multiset_out += v[0] * v[1]
            output += multiset_out
        return output
    
class Identity():
    def forward(x):
        return 1
    def __str__(self):
        return "Identity"

