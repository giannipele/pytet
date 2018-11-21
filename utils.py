
def tokens_substr(tetstr, tokens='{}'):
    """Return the substring between the two parenthesis. The function copies
    the part of the string between the first occurrence of the opening 
    character and the respective closing one.

    Example:
        Given a string such as "123{content}", the function returns 
        "content". 

    Args:
        tetstr (str): Full string.
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
    return substr[1:]
