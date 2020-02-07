"""
Practice > Algorithms > Strings > Sherlock and the Valid String

Sherlock considers a string to be valid if all characters of the string appear the same number of times.
It is also valid if he can remove just 1 character at 1 index in the string, and the remaining characters
will occur the same number of times. Given a string 's', determine if it is valid.
If so, return YES, otherwise return NO.

For example, if s = abc, it is a valid string because frequencies are {a:1, b:1, c:1}.
So is 's = abcc' because we can remove one c and have 1 of each character in the remaining string.
If s = abccc however, the string is not valid as we can only remove 1 occurrence of c.
That would leave character frequencies of {a:1, b:1, c:2}.

Note: The removal of all instances of a character is considered 1 removal,
not the number of times any character is removed
"""

import os
from time import time

TESTS = [("abc", "YES"), ("abcc", "YES"), ("abccc", "NO"), ("aabbcd", "NO"), ("aabbccddeefghi", "NO"),
         ("abcdefghhgfedecba", "YES"), ("aabbc", "YES"), ("aab", "YES"), ("aaabbccc", "NO")]


def isValid(s):
    s = list(s)
    unique = set(s)
    count = sorted([s.count(e) for e in unique])
    if len(count) == 1:
        return "YES"
    elif count[0] == 1 and all([k == count[1] for k in count[1:]]):
        return "YES"
    else:
        m = count[0]
        changes = 0
        for k in count[1:]:
            if k > m + 1:
                return "NO"
            if k == m + 1:
                changes += 1
        if changes > 1:
            return "NO"
        else:
            return "YES"


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = isValid(s)
    fptr.write(result + '\n')
    fptr.close()


def test():
    data = TESTS
    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = isValid(input)
        t1 = time() - t0
        if not rst == output:
            print(f"{input} Expected: {output} Result: {rst}")
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
