"""
Practice > Algorithms > Strings > How many Substrings

Consider a string of n characters, s, of where each character is indexed from 0 to n-1.

You are given 'q' queries in the form of two integer indices: left and right.
For each query, count and print the number of different substrings of s in the inclusive range between left and right.

Note: Two substrings are different if their sequence of characters differs by at least one.
For example, given the string  s=aab, substrings  s[0,0]=a and  s[1,1]=a are the same but
substrings  s[0,1]=aa  and  s[1,2]=ab  are different.
"""

import sys
import os
from time import time


TESTS = [("aabaa", [[1, 4]], [8]),
         ("aabaa", [[1, 1], [1, 4], [1, 1], [1,4], [0,2]], [1, 8, 1, 8, 5])]


def distinctSubstring(s):
    # Put all distinct substring in a HashSet
    # This solves the count by brute force....
    result = set()
    acc = []
    prev = 0

    # List All Substrings
    for i in range(len(s) + 1):
        for j in range(i + 1, len(s) + 1):
            # Add each substring in Set
            result.add(s[i:j]);
            # Return size of the HashSet
            acc.append(len(result))
            prev = acc[-1]
    return len(result)


def countSubstrings(s, queries):
    result = []
    for query in queries:
        result.append(distinctSubstring(s[query[0]:query[1]+1]))
    return result


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nq = input().split()
    n = int(nq[0])
    q = int(nq[1])
    s = input()
    queries = []

    for _ in range(q):
        queries.append(list(map(int, input().rstrip().split())))

    result = countSubstrings(s, queries)
    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')
    fptr.close()


def test():
    data = TESTS
    for d in data:
        input_str = d[0]
        input_query = d[1]
        output = d[2]

        t0 = time()
        rst = countSubstrings(input_str, input_query)
        t1 = time() - t0
        if not rst == output:
            print(f"{input} Expected: {output} Result: {rst}")
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
