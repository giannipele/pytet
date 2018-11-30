from anytree import NodeMixin 
import autograd.numpy as np

class ComputationalNode():
    def __init__(self, params, leaf):
        self.params = params
        self.leaf = leaf
        self.multisets = []

    def add_multiset(self):
        self.multisets.append([])
        return len(self.multisets)

    def add_value_to_multiset(self, i, n):
        self.multisets[i].append(n)

    def count_nodes(self):
        count = 1
        for m in self.multisets:
            for v in m:
                sub_node = v.count_nodes()
                count += sub_node
        return count


    def forward(self):
        #print("BOOOO")
        #print("PAR: ", self.params)
        if self.leaf:
            return 1
        else:
            output = self.params[0]
            for i, m in enumerate(self.multisets):
                multiset_out = np.zeros(len(m))
                for j, v in enumerate(m):
                    multiset_out[j] = self.params[i+1] * v.forward()
                #print("PRESIG: ", multiset_out)
                output += np.sum(multiset_out)
            return self.sigmoid(output)
    
    def sigmoid(self,x):
        return 1/ (1+np.exp(-x)) 
    
    
