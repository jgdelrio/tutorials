"""
Practice > Algorithms > Strings > Two Characers

In this challenge, you will be given a string. You must remove characters until the string is
made up of any two alternating characters. When you choose a character to remove, all instances
of that character must be removed. Your goal is to create the longest string possible that
contains just two alternating letters.

As an example, consider the string abaacdabd. If you delete the character a,
you will be left with the string bcdbd. Now, removing the character c
leaves you with a valid string bdbd having a length of 4. Removing either b or d
at any point would not result in a valid string.

Given a string 's', convert it to the longest possible string 't' made up only of alternating characters.
Print the length of string 't' on a new line. If no string 't' can be formed, print 0 instead.
"""

import os
from time import time

TESTS = [("beabeefeab", 5)]


def validation(array):
    for i in range(len(array) - 1):
        if array[i] == array[i + 1]:
            return False
    return True


def alternate(s):
    s = s.strip()

    characters = list(set(s))
    n_chars = len(characters)
    max_len = 0
    for x in range(n_chars):
        for y in range(x + 1, n_chars):
            # Create strings with unique combinations of pairs of characters within the string
            candidate = [c for c in s if c == characters[x] or c == characters[y]]

            # Validate if the new candidate is alternating characters
            if validation(candidate):
                max_len = max(max_len, len(candidate))
    return max_len


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    l = int(input().strip())
    s = input()
    result = alternate(s)
    fptr.write(str(result) + '\n')
    fptr.close()


def test():
    data = TESTS

    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = alternate(input)
        t1 = time() - t0
        assert (rst == output)
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
