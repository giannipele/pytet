
def next_token(string, token, index=0, parenthesis=False):
    substr = ""
    folding = False
    p_checker = 0
    while string[index] != token or folding:
        substr += string[index]
        if parenthesis:
            if string[index] == '(':
                p_checker += 1
            elif string[index] == ')':
                p_checker -= 1
            if p_checker == 0:
                folding = False
            else:
                folding = True
        index += 1
    return substr, index


class TetValue:
    def __init__(self, valuestr=""):
        self.multisets = []
        if valuestr != "":
            self.parse_value_str(valuestr, 0)

    def __str__(self):
        if len(self.multisets) == 0:
            return str(self.top)
        else:
            stringify = '('
            stringify += str(self.top)
            for m in self.multisets:
                stringify += ",{}".format(m.__str__())
            stringify += ')'
            return stringify

    def parse_value_str(self, valuestr, index):
        try:
            if valuestr[index] == '(':
                index += 1
                self.top = valuestr[index]
                #print("Top: {}".format(index))
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



class TetMultiset: 
    def __init__(self, valuestr=""): 
        self.elements = []

    def __str__(self):
        if len(self.elements) == 0:
            return "[ ]"
        else:
            stringify = "["
            for e in self.elements:
                stringify += "{}:{},".format(e[0].__str__(), e[1])
            stringify = stringify[:-1] + ']'
            return stringify

    def parse_multiset_str(self, valuestr, index):
        try: 
            if valuestr[index] != '[':
                raise Exception("Malformed string. Expected '[', found '{0}' at position {1}".format(valuestr[index], index)) 
            while valuestr[index] != ']': 
                #print("Multiset idx: {}".format(index)) 
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
                #print(count)
                self.elements.append((value, int_count)) 
            return index + 1
        except Exception as e:
            print('Exception: {}'.format(e))
            

value = TetValue()
index = 0
index = value.parse_value_str("(T,[(T,[T:4]):3,(T,[T:2]):1],[(T,[]):1,(T,[T:8]):6,(T,[]):2 ])", 0)
print(value.count_nodes())
v = str(value)
print(v)
