
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

def parse_function(string):
    s_split = string.split(',')
    params = []
    for p in s_split[1:]:
        params.append(float(p))
    return s_split[0], params


class TetValue:
    def __init__(self, valuestr=""):
        self.multisets = []
        if valuestr != "":
            self.parse_value(valuestr, 0)

    def parse_value(self, valuestr, index):
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

    def parse_multiset_str(self, valuestr, index):
        try:
            if valuestr[index] != '[':
                raise Exception("Malformed string. Expected '[', found '{0}' at position {1}".format(valuestr[index], index))
            while valuestr[index] != ']':
                #print("Multiset idx: {}".format(index))
                value = TetValue()
                index = value.parse_value(valuestr, index + 1)
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

class RnnTet:
    def __init__(self, tetstr=""):
        self.type = ""
        self.children = []
        if tetstr != "":
            self.parse_tet_str(tetstr, 0)
    
    def parse_tet_str(self, tetstr, index=0):
        #print("INDEX: {}".format(index))
        #tetstr = tetstr.strip()
        tetstr = tetstr.replace(' ','')
        #print(tetstr[index:] + "\n")
        if tetstr[index] != '{':
            #print("String: {}, Index:{}".format(tetstr,index))
            raise Exception("Malformed string. Expected '{', found '{0}' at position {1}".format(tetstr[index], index))
        
        index += 1
        substr, index = next_token(tetstr, '{', index) 
        if substr != "NODE":
            #print("String: {}, Index:{}".format(tetstr,index))
            raise Exception("Malformed string. Expected 'NODE', found '{0}'".format(substr))
        
        index += 1
        substr, index = next_token(tetstr, '(', index) 
        if substr != "FUNCTION":
            #print("String: {}, Index:{}".format(tetstr,index))
            raise Exception("Malformed string. Expected 'FUNCTION', found '{0}'".format(substr))
       
        index += 1
        substr, index = next_token(tetstr, ')', index)
        self.fun, self.params = parse_function(substr)
        
        index += 3
        substr, index = next_token(tetstr, '(', index)
        if substr != "TYPE":
            #print("String: {}, Index:{}".format(tetstr,index))
            raise Exception("Malformed string. Expected 'TYPE', found '{0}'".format(substr))

        index += 1
        substr, index = next_token(tetstr, ')', index, parenthesis=True)
        self.type = substr

        index += 2
        while tetstr[index] != '}':
            if tetstr[index] == '{':
                index += 1
                substr, index = next_token(tetstr, '(', index)
                if substr != "CHILD":
                    #print("String: {}, Index:{}".format(tetstr,index))
                    raise Exception("Malformed string. Expected 'CHILD', found '{0}'".format(substr))

                substr, index = next_token(tetstr, ')', index + 1)
                child = RnnTet()
                index = child.parse_tet_str(tetstr, index + 1)
                self.children.append(child)
            index += 2 
        return index
    
    def print_tet(self, indent=0):
        prefix = "\t" * indent
        print("\t"*indent + "{NODE")
        #indent += 1
        print("\t"*(indent+1) + "FUNCTION: {}, {}".format(self.fun, self.params))
        print("\t"*(indent+1) + "TYPE: {}".format(self.type))
        print("\t"*(indent+1) + "CHILDREN [")
        for c in self.children:
            c.print_tet(indent + 2)

        print("\t"*(indent+1) + "]")
        print("\t"*indent + "}")


value = TetValue()
index = 0
index = value.parse_value("(T,[(T,[T:4]):3,(T,[T:2]):1],[(T,[]):1,(T,[T:8]):6])", 0)
#print("Index: {}".format(index))
#print("Number of nodes: {}".format(value.count_nodes()))
tet = RnnTet ("{NODE {FUNCTION (logistic,-75,10)}{TYPE ()}}")
try:
    print(tet.parse_tet_str("{NODE {FUNCTION (logistic,-75,10)}{TYPE ()}{CHILD (paper0) {NODE {FUNCTION (logistic,-75,10)}{TYPE (author_paper(author0,paper0))}{CHILD (paper1) {NODE {FUNCTION (identity)}{TYPE (paper_paper(paper1,paper0))}}}}}}", 0))
except Exception as e:
    print(e)
tet.print_tet()
print(tet.parse_tet_str("{NODE {FUNCTION (logistic,-75,10)}{TYPE ()}{CHILD (paper0) {NODE {FUNCTION (logistic,-75,10)}{TYPE (author_paper(author0,paper0))}{CHILD (paper1) {NODE {FUNCTION (identity)}{TYPE (paper_paper(paper1,paper0))}}}{CHILD (paper1) {NODE {FUNCTION (identity)}{TYPE (paper_paper(paper1,paper0))}}}}}}}"))
tet.print_tet()
