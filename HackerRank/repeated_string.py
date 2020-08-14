"""
Practice > Algorithms > Implementation > Repeated String

Lilah has a string, s, of lowercase English letters that she repeated infinitely many times.

Given an integer, n, find and print the number of letter a's in the first n letters of Lilah's infinite string.

For example, if the string s='abcac' and n=10, the substring we consider is abcacabcac,
the first 10 characters of her infinite string. There are 4 occurrences of a in the substring.

Function Description
--------------------
Complete the repeatedString function in the editor below. It should return an integer representing the number of occurrences of a in the prefix of length  in the infinitely repeating string.

repeatedString has the following parameter(s):

s: a string to repeat
n: the number of characters to consider

"""

import os
from time import time

TESTS = [[('aba', 10), 7],
         [('a', 1000000000000), 1000000000000]]


def repeatedString_v0(s, n):
    number_of_a = s.count('a')
    length = (number_of_a)*(n//len(s))
    remainder = (s[:n % len(s)].count("a"))
    return (length + remainder)


def repeatedString(s, n):
    return ((s.count('a'))*(n//len(s)) + (s[:n % len(s)].count("a")))


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    n = int(input())
    result = repeatedString(s, n)
    fptr.write(str(result) + '\n')
    fptr.close()


def test():
    data = TESTS
    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = repeatedString(*input)
        t1 = time() - t0
        if not rst == output:
            print(f"{input} Expected: {output} Result: {rst}")
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
