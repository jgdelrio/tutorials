"""
Practice > Algorithms > Strings > Funny String

In this challenge, you will determine whether a string is funny or not.
To determine whether a string is funny, create a copy of the string in reverse e.g. abc -> cba.
Iterating through each string, compare the absolute difference in the ascii values of the characters
at positions 0 and 1, 1 and 2 and so on to the end. If the list of absolute differences
is the same for both strings, they are funny.

Determine whether a give string is funny. If it is, return Funny, otherwise return Not Funny.

For example, given the string "s = lmnop", the ordinal values of the charcters
are [108, 109, 110, 111, 112]. "s = ponml" and the ordinals are [112, 111, 110, 109, 108].
The absolute differences of the adjacent elements for both strings are [1, 1, 1, 1], so the answer is Funny.
"""

import os
from time import time

TESTS = [("acxz", "Funny"), ("bcxz", "Not Funny")]


def funnyString(s):
    as_number = list(map(ord, s))
    diff = []
    base = as_number[0]
    for ind in range(1, len(as_number)):
        diff.append(abs(as_number[ind] - base))
        base = as_number[ind]

    if diff == list(reversed(diff)):
        return "Funny"
    else:
        return "Not Funny"


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(input())

    for q_itr in range(q):
        s = input()
        result = funnyString(s)
        fptr.write(result + '\n')

    fptr.close()


def test():
    data = TESTS
    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = funnyString(input)
        t1 = time() - t0
        if not rst == output:
            print(f"{input} Expected: {output} Result: {rst}")
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
