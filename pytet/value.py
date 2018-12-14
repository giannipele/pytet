import autograd.numpy as np

class TetValue:
    """
    Class that reoresents a TetValue, a nested multisets of multisets structure
    of boolean assignment.

    Args:
        valuestr    (str) : String value of the TetValue.

    Attributes:
        top         (str) : Top value of the TetValue.
        multisets   (:obj:'list') : List of the children multiset.
    """
    def __init__(self, valuestr=""):
        self.multisets = []
        if valuestr != "":
            self.parse_value_str(valuestr, 0)

    def __str__(self):
        """Stringify the TetValue. Same representation as the one for the
           init valuestr input."""
        if len(self.multisets) == 0:
            return str(self.top)
        else:
            stringify = '('
            stringify += str(self.top)
            for m in self.multisets:
                stringify += ",{}".format(m.__str__())
            stringify += ')'
            return stringify

    def parse_value_str(self, valuestr, index=0):
        """Take in input a TetValue string, recursively generate the value."""
        try:
            if valuestr[index] == '(':
                index += 1
                self.top = valuestr[index]
                index += 1
                while valuestr[index] != ')':
                    multiset = TetMultiset()
                    index = multiset.parse_multiset_str(valuestr, index + 1)
                    self.multisets.append(multiset)
                return index + 1
            elif valuestr[index] == ']':
                return index
            else:
                self.top = valuestr[index]
                return index + 1
        except Exception as e:
            print("Exception: {}".format(e))

    def count_nodes(self):
        count = 1
        for m in self.multisets:
            for e in m.elements:
                sub_nodes = e[0].count_nodes()
                count += sub_nodes * e[1]
        return count

    def arrayfy(self):
        if len(self.multisets) == 0:
            return 1
        else:
            v_array = []
            for i, m in enumerate(self.multisets):
                v_array.append([])
                for v in m.elements:
                    v_array[i].append([v[0].arrayfy(), v[1]])
            return v_array

#    def arrayfy_2(self):
#        if len(self.multisets) == 0:
#            return [np.float(1)]
#        else:
#            v_array = []
#            for i, m in enumerate(self.multisets):
#                v_array.append([])
#                for v in m.elements:
#                    v_array[i].append((v[0].arrayfy_2(), v[1]))
#            return v_array

class TetMultiset:
    """
    Multiset class contains the TetValues and their multiplcity.

    Args:
        valuestr    (str) : Input string of the multiset representation.

    Attributes
        elements    (:obj:'list') : List of tuples of (TetValue, count)
    """
    def __init__(self, valuestr=""): 
        self.elements = []

    def __str__(self):
        """Stringify the Multiset."""
        if len(self.elements) == 0:
            return "[ ]"
        else:
            stringify = "["
            for e in self.elements:
                stringify += "{}:{},".format(e[0].__str__(), e[1])
            stringify = stringify[:-1] + ']'
            return stringify

    def parse_multiset_str(self, valuestr, index):
        """Generate the multiset from the string representation."""
        if valuestr[index] != '[':
            raise Exception("Malformed string. Expected '[', found '{0}' at position {1}".format(valuestr[index], index)) 
        while valuestr[index] != ']': 
            value = TetValue()
            index = value.parse_value_str(valuestr, index + 1)
            if valuestr[index] == ']':
                break
            elif valuestr[index] != ':': 
                raise Exception("Malformed string. Expected ':', found '{0}' at position {1}".format(valuestr[index], index))
            index += 1 
            count = "" 
            while valuestr[index] != ',' and valuestr[index] != ']':
                count += valuestr[index] 
                index += 1 
            int_count = int(count)
            self.elements.append((value, int_count)) 
        return index + 1



