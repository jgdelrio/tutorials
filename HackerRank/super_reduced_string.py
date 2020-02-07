"""
Practice > Algorithms > Strings > Super Reduced String

Steve has a string of lowercase characters in range ascii[‘a’..’z’]. He wants to reduce the string
to its shortest length by doing a series of operations. In each operation he selects a pair of
adjacent lowercase letters that match, and he deletes them. For instance, the string aab could be
shortened to b in one operation.

Steve’s task is to delete as many characters as possible using this method and print the resulting string.
If the final string is empty, print Empty String

Function Description
--------------------
Complete the superReducedString function in the editor below. It should return the super reduced string
or Empty String if the final string is empty.

superReducedString has the following parameter(s):

s: a string to reduce
"""

import os
from time import time

TESTS = [("aaabccddd", "abd"), ("aa", "Empty String")]


def reduce_string(s):
    s = list(s)
    ind = len(s) - 1
    reduced = False
    while ind > 0:
        if s[ind] == s[ind-1]:
            s.pop(ind)
            s.pop(ind-1)
            ind -= 2
            reduced = True
        else:
            ind -= 1
    if s:
        return "".join(s), reduced
    else:
        return "Empty String", False


def superReducedString(s):
    reduced = True
    while reduced:
        s, reduced = reduce_string(s)
    return s


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = superReducedString(s)
    fptr.write(result + '\n')
    fptr.close()


def test():
    data = TESTS
    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = superReducedString(input)
        t1 = time() - t0
        assert (rst == output)
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
