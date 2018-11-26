import autograd.numpy as np
import re
from anytree import NodeMixin, RenderTree
from functions import Logistic, Identity
from utils import tokens_substr
from tetutils import TetValue

class RnnTet(NodeMixin):
    """
    Documentation: to be done.
    """
    def __init__(self, tetstring = "", binds="", parent=None):
        super(RnnTet, self).__init__()
        if tetstring != "":
            self.parent=parent
            self.binds = binds
            self.parse_tet_str(tetstring)

    def __str__(self):
        render = ""
        for pre, fill, node in RenderTree(self):
            render = render +"{}{}\n".format(pre, node.name)
        return render

    def parse_tet_str(self, tetstr):
        regex = re.compile(r'[\n\t\r]')
        tetstr = regex.sub(" ", tetstr)
        tetstr = tetstr.replace(' ','')
        index = 0
        if tetstr[index] != '{':
            raise Exception("Malformed string. Expected '{', found '{}' at position {}".format(tetstr[index], index))

        index += 1
        substr = tetstr[index:]
        if not substr.startswith("NODE"):
            raise Exception("Malformed string. Expected 'NODE', found '{0}'".format(substr))

        index += 4
        substr = tokens_substr(tetstr[index:])
        if not substr.startswith("FUNCTION"):
            raise Exception("Malformed string. Expected 'FUNCTION', found '{0}'".format(substr))
        else:
            function = tokens_substr(substr,'()')
            self.__parse_activation_function(function)
        
        index += len(substr)+2
        substr = tokens_substr(tetstr[index:]) 
        if not substr.startswith("TYPE"):
            raise Exception("Malformed string. Expected 'TYPE', found '{0}'".format(substr))
        else:
            #self.rtype = tokens_substr(substr,'()')
            self.name = tokens_substr(substr,'()')

        index += len(substr)+2
        while tetstr[index] != '}':
            substr = tokens_substr(tetstr[index:])
            if not substr.startswith("CHILD"):
                raise Exception("Malformed string. Expected 'CHILD', found '{0}'".format(substr))
            else:
                variables = tokens_substr(substr, '()')
                bind_variables = variables.split(',')
                subtet = tokens_substr(substr)
                child = RnnTet(tetstring ="{"+subtet+"}", parent=self, binds=bind_variables)
                index += len(substr)+2
        return True

    def __parse_activation_function(self,string):
        s_split = string.split(',')
        params = [float(p) for p in s_split[1:]]
        s_split[0], params
        if s_split[0] == 'logistic':
            self.activation = Logistic(params)
        elif s_split[0] == 'identity':
           self.activation = Identity()

    def print_tet(self, indent=0):
        prefix = "\t" * indent
        print("\t"*indent + "{NODE")
        print("\t"*(indent+1) + "BIND VARS: {}".format(self.binds))
        print("\t"*(indent+1) + "FUNCTION: {}".format(self.activation))
        print("\t"*(indent+1) + "TYPE: {}".format(self.name))
        print("\t"*(indent+1) + "CHILDREN [")
        for c in self.children:
            c.print_tet(indent + 2)

        print("\t"*(indent+1) + "]")
        print("\t"*indent + "}")

    def compute_value(self, value):
        if self.is_leaf:
            return self.activation.forward()
        else:
            computations = []
            for i, child in enumerate(self.children):
                multiset = value.multisets[i]
                computations.append([])
                for v in multiset.elements:
                    computations[i].append([child.compute_value(v[0]), v[1]])
            comp = np.asarray(computations)
            s = self.activation.forward(comp)
            return s

file = open('tet-a.verbose', 'r')
tet_txt = file.read()
file.close()

node = RnnTet(tet_txt)
#node.print_tet()

value = TetValue()
index = value.parse_value("(T,[(T,[T:8]):4,(T,[T:9]):2,(T,[T:10]):2])", 0)

c = node.compute_value(value)
print(c)
