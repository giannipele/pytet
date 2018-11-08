#from tetutils import RnnTet
#import torch
#
#
#def ComputationalTemplate():
#    def __init__(self, tet):
#        self.variables = []
#        self.children = []
#        self.function = tet.fun
#        for p in tet.params:
#            variables.append(torch.tensor(p, requires_grad=True))
#        for c in tet.children:
#            l = ComputationalTemplate(c)
#            self.children.append(l)
#
#
#
#def ComputationalGraph():
#    def forward(self, tetvalue, templatenode):
#        if len(templatenode.children) == 0:
#            return 1
#        else:
#            evaluation = torch.tensortemplateNode.params[0]
#            for m in tetvalue.multisets:
#                s = 0
#                for e in m.elements:
#                    s += 

        
import autograd.numpy as np
from autograd import grad

class LogisticEvaluationFunction():
    def forward(params, tetvalue):
        
