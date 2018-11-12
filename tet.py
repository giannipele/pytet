from anytree import NodeMixin

def tokens_substr(tetstr, tokens=('{','}')):    
    """Return the substring between the two parenthesis. tetstr
    first character must be the first one in tokens[].

    Args:
        tetstr (str): Full string.
        token (tuple): Opening and closing character that include the substring.

    Returns:
        str: The substring between the two tokens.
    """
    substr = ""
    folding = False
    index = 0
    checker = 0
    while not folding and index < len(tetstr):
        substr += tetstr[index]
        if tetstr[index] == tokens[0]:
            checker += 1
        elif tetstr[index] == tokens[1]:
            checker -= 1
        if checker == 0:
            folding = True
        index += 1
    return substr

class Tet(NodeMixin):
    def __init__(self, tetstring = ""):
        super(Tet, self).__init__()
        if tetstr != "":
            self.str_to_value(tetstr)

    def str_to_value(self, tetstr):
        tetstr = tetstr.replace(' ','')
        index = 0
        if tetstr[index] != '{':
            raise Exception("Malformed string. Expected '{', found '{0}' at position {1}".format(tetstr[index], index))

        index += 1
        substr = next_token(tetstr[index:]) 
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
        fun, params = parse_function(substr)
        if fun == 'logistic':
            self.activation = Logistic(params)
        elif fun == 'identity':
           self.activation = Identity()
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


n = token_substr("{{ciao}", ('{','}')) 
print(n)
