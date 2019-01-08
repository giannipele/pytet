from value import TetValue

def tokens_substr(tetstr, tokens='{}'):
    """Return the substring between the two tokens. The function copies
    the part of the string between the first occurrence of the opening 
    character and the respective closing one.

    Example:
        Given a string such as "123{hello}", the function returns 
        "hello". With nested recurrences as in "[([hello])]" the function
        returns "([hello)]".

    Args:
        tetstr (str): The string to parse.
        token (str): Opening and closing character that include the substring.

    Returns:
        str: The substring between the two tokens.
    """
    substr = ""
    folding = False
    copy = False
    index = 0
    checker = 0
    while not folding and index < len(tetstr):
        if tetstr[index] == tokens[0]:
            if checker == 0:
                copy = True
            checker += 1
        elif tetstr[index] == tokens[1]:
            checker -= 1
        if checker == 0 and copy == True:
            folding = True
            copy = False
        if copy:
            substr += tetstr[index]
        index += 1
    if substr == "":
        return ""
    else:
        return substr[1:]

def read_values_file(file_path):
    values = []
    with open(file_path, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            v = TetValue(line)
            #values.append(v.arrayfy())
            values.append(v)
    return values

def read_labels_file(file_path):
    labels = []
    with open(file_path, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            labels.append(int(line))
    return labels

def read_values_labels_files(values_path, labels_path):
    return read_values_file(values_path), read_labels_file(labels_path)

