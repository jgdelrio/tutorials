"""
Practice > Algorithms > Strings > CamelCase

Alice wrote a sequence of words in CamelCase as a string of letters, , having the following properties:

It is a concatenation of one or more words consisting of English letters.
All letters in the first word are lowercase.
For each of the subsequent words, the first letter is uppercase and rest of the letters are lowercase.
Given s, print the number of words in  on a new line.

For example, s = oneTwoThree. There are 3 words in the string.
"""

import os
from time import time

TESTS = [("saveChangesInTheEditor", 5), ("other", 1)]


def camelcase(s):
    s = list(s)
    is_upper = [k.isupper() for k in s]
    return sum(is_upper) + 1


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = camelcase(s)
    fptr.write(str(result) + '\n')
    fptr.close()


def test():
    data = TESTS
    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = camelcase(input)
        t1 = time() - t0
        if not rst == output:
            print(f"{input} Expected: {output} Result: {rst}")
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
