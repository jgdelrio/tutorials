"""
Practice > Algorithms > Strings > Making Anagrams

We consider two strings to be anagrams of each other if the first string's letters can be rearranged
to form the second string. In other words, both strings must contain the same exact letters in the
same exact frequency. For example, bacdc and dcbac are anagrams, but bacdc and dcbad are not.

Alice is taking a cryptography class and finding anagrams to be very useful. She decides on an encryption scheme involving two large strings where encryption is dependent on the minimum number of character deletions required to make the two strings anagrams. Can you help her find this number?

Given two strings, s1 and s2, that may not be of the same length, determine the minimum number
of character deletions required to make s1 and s2 anagrams.
Any characters can be deleted from either of the strings.

For example, s1=abc and s2=amnot. The only characters that match are the a's so we have to remove bc from s1
and mnot from s2 for a total of 6 deletions.
"""

import os
from time import time

TESTS = [("cde", "abc", 4), ("abc", "amnop", 6)]


def makingAnagrams(s1, s2):
    unique_s1 = set(s1)
    unique_s2 = set(s2)
    count_s1 = {e: s1.count(e) for e in unique_s1}
    count_s2 = {e: s2.count(e) for e in unique_s2}

    deletions = 0
    for s in unique_s1:
        if s in unique_s2:
            deletions += abs(count_s1[s] - count_s2[s])
        else:
            deletions += count_s1[s]
    for s in unique_s2:
        if s not in unique_s1:
            deletions += count_s2[s]
    return deletions


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s1 = input()
    s2 = input()
    result = makingAnagrams(s1, s2)
    fptr.write(str(result) + '\n')
    fptr.close()


def test():
    data = TESTS
    for d in data:
        s1 = d[0]
        s2 = d[1]
        output = d[2]

        t0 = time()
        rst = makingAnagrams(s1, s2)
        t1 = time() - t0
        if not rst == output:
            print(f"{input} Expected: {output} Result: {rst}")
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
