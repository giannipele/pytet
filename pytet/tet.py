import autograd.numpy as np
import re
from anytree import NodeMixin, RenderTree
from functions import Logistic, Identity
from utils import tokens_substr
from value import TetValue, TetMultiset

class RnnTet(NodeMixin):
    """
    A node in the RnnTet.

    Args:
        tetstr      (str,optional): RnnTet string description.
                                    Default parser is 'verbose'.
        binds       (:obj:'list',optional): List of variables string name.
                                            Default is [].
        parent      (:obj:'anytree.NodeMixin', optional): Parent node. If parent is missing,
                                                          this node is the root.

    Attributes:
        parent      (:obj:'anytree.NodeMixin') : Parent node of this node.
        binds       (:obj:'list') : List of binding variables for this node.
        name        (str) : String name of the node, name represents the
                     logical predicate of the TET.
        children    (:obj:'list') : List of children nodes.
        activation  (:obj:'functions') : Function object of the node.

    """
    def __init__(self, tetstring = "", binds=[], parent=None):
        super(RnnTet, self).__init__()
        self.parent=parent
        self.binds = binds
        if tetstring != "":
            self.parse_tet_str(tetstring)

    def __str__(self):
        """Default method of the anytree library to stringify the TET."""
        render = ""
        for pre, fill, node in RenderTree(self):
            binds = ""
            for b in node.binds:
                binds += str(b) + ','
            render = render +"{}{} {}\n".format(pre,binds, node.name)
        return render

    def parse_tet_str(self, tetstr, parser='v'):
        """Function that decide which parser to use for the
           RnnTet string representation. Actual parser are
           'v' = verbose
           'c' = compact
        """
        if parser == 'v':
            self.__parse_tet_verbose(tetstr)
        elif parser == 'c':
            self.__parse_tet_compact(tetstr)
        else:
            print("Unkown value for parser flag. Values are 'v' = verbose, 'c' = compact.")
            return

    def __parse_tet_verbose(self, tetstr):
        """Parser for the verbose TET representation."""
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
            self.name = tokens_substr(substr,'()')
            if self.name == "":
                self.name = "T()"

        index += len(substr)+2
        while tetstr[index] != '}':
            substr = tokens_substr(tetstr[index:])
            if not substr.startswith("CHILD"):
                raise Exception("Malformed string. Expected 'CHILD', found '{0}'".format(substr))
            else:
                variables = tokens_substr(substr, '()')
                bind_variables = variables.split(',')
                subtet = tokens_substr(substr)
                child = RnnTet(parent=self, binds=bind_variables)
                child.parse_tet_str("{"+subtet+"}", parser='v' )
                index += len(substr)+2
        return True

    def __parse_tet_compact(self,tetstr):
        """Parser for the compact TET representation."""
        regex = re.compile(r'[\n\t\r]')
        tetstr = regex.sub(" ", tetstr)
        tetstr = tetstr.replace(' ','')
        index = 0
        if tetstr[index] != '(':
            raise Exception("Malformed string. Expected '(', found '{0}' in sequence ...{1}...".format(tetstr[index],tetstr[index-4:]))

        substr = tokens_substr(tetstr[index:], tokens='()')
        self.__parse_activation_function(substr)

        index += len(substr)+2
        if tetstr[index] != '[':
            raise Exception("Malformed string. Expected '[', found '{0}'".format(tetstr[index]))

        index += 1
        name = tokens_substr(tetstr[index:],'()')
        if name == "":
            self.name = "T()"
        else:
            self.name = name
        index += len(name) + 2
        if tetstr[index] == ']':
            return True

        index += 1
        while tetstr[index] != ']' and index < len(tetstr) :

            substr = tokens_substr(tetstr[index:], '[]')
            variables = tokens_substr(tetstr[index:], '()')
            bind_variables = variables.split(',')

            tmp_index = len(variables) + 2
            if substr[tmp_index] == ']':
                raise Exception("Malformed string. Expected ',', found '{0}'".format(tetstr[tmp_index:]))

            child = RnnTet(parent=self, binds=bind_variables)
            child.parse_tet_str(substr[tmp_index+1:], parser='c' )

            index += len(substr) + 2
        return True


    def __parse_activation_function(self,string):
        """Parse the string contatining the information of the
           node's activation function,and initialize node's
           variable <activation>"""
        s_split = string.split(',')
        params = [float(p) for p in s_split[1:]]
        s_split[0], params
        if s_split[0] == 'logistic':
            self.activation = Logistic(params)
        elif s_split[0] == 'identity':
            self.activation = Identity()

    def print_tet(self, indent=0):
        """Ad-Hoc indented printing method of the RnnTet.
           Current information printed:
                Binding Variables
                Activation Function
                Name (Type) of the node
                Children-TET nodes"""
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

    def get_params(self):
        if self.is_leaf:
            return self.activation.params
        else:
            params = [_ for _ in self.activation.params]
            for child in self.children:
                params.append(child.get_params())
            return params

    def forward_value(self, params, value, eval_values=None):
        if self.is_leaf:
            r = self.activation.forward()
            if eval_values is not None:
                eval_values.top = r
            return r
        else:
            multisets = []
            multisets_out = []

            for i, m in enumerate(value):
                if eval_values is not None:
                    multisets.append(TetMultiset())
                child = self.children[i]
                multiset_out = []

                for v in m:
                    if eval_values is not None:
                        sub_value = TetValue()
                        multiset_out.append([child.forward_value(params[i+len(value)+1],
                            v[0], sub_value), v[1]])
                        multisets[i].elements.append((sub_value, v[1]))
                    else:
                        multiset_out.append([child.forward_value(params[i+len(value)+1],
                            v[0]), v[1]])
                multisets_out.append(multiset_out)
            r = self.activation.forward(params, multisets_out)
            if eval_values is not None:
                eval_values.top = r
                eval_values.multisets = multisets
            return r


def loss(par, value, tet, evaluations=None):
    return ((tet.forward_value(par, value, evaluations) - 1)**2)/2


