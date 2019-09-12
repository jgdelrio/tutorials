"""
Algorithms > Implementation > Bigger is Greater

Lexicographical order is often known as alphabetical order when dealing with strings.
A string is greater than another string if it comes later in a lexicographically sorted list.
Given a word, create a new word by swapping some or all of its characters.
This new word must meet two criteria:
  1) It must be greater than the original word
  2) It must be the smallest word that meets the first condition

For example, given the word   w = abcd   the next largest word is   abdc.

Complete the function biggerIsGreater below to create and return the new string meeting the criteria.
If it is not possible, return no answer.

biggerIsGreater has the following parameter(s):
  w: a string
  Input Format

Constraints:
  1 <= T <= 1e6
  1 <= |w| <= 100
  w will contain only letters in the range ascii[a..z]

Output Format:
  For each test case, output the string meeting the criteria. If no answer exists, print no answer.
"""

import os
from time import time

TEST1 = (['ab', 'ba'],
         ['bb', 'no answer'],
         ['hefg', 'hegf'],
         ['dhck', 'dhkc'],
         ['dkhc', 'hcdk'])


def biggerIsGreater(word):
    # Find non-increasing suffix
    idx = len(word) - 1
    while idx > 0 and word[idx - 1] >= word[idx]:
        idx -= 1
    if idx <= 0:
        return 'no answer'

    # Find successor to pivot
    j = len(word) - 1
    while word[j] <= word[idx - 1]:
        j -= 1
    word[idx - 1], word[j] = word[j], word[idx - 1]

    # Reverse suffix
    word[idx:] = word[len(word) - 1: idx - 1: -1]
    return ''.join(word)


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    T = int(input())

    for T_itr in range(T):
        w = input()
        result = biggerIsGreater(list(w))
        fptr.write(result + '\n')

    fptr.close()


def test():
    data = TEST1

    for d in data:
        input = d[0]
        output = d[1]

        rst = biggerIsGreater(list(input))
        assert (rst == output)


if __name__ == '__main__':
    test()




