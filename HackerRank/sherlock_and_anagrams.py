"""
Practice > Algorithms > Strings > Sherlock adn Anagrams

Two strings are anagrams of each other if the letters of one string can be rearranged
to form the other string. Given a string, find the number of pairs of substrings of the
string that are anagrams of each other.

For example s=mom, the list of all anagrammatic pairs is [m,m], [mo,om] at positions
[[0],[2]], [[0,1],[1,2]] respectively.

Function Description
--------------------
Complete the function sherlockAndAnagrams in the editor below.
It must return an integer that represents the number of anagrammatic pairs of substrings in s.

sherlockAndAnagrams has the following parameter(s):
    s: a string .

"""

import os
from time import time
from collections import Counter
from itertools import combinations


TESTS = [
    ['abba', 4],
    ['abcd', 0],
    ['ifailuhkqq', 3],
    ['kkkk', 10],
]


def sherlockAndAnagrams(s: str) -> int:
    count = []
    n = len(s)
    for i in range(1, n+1):
        a = ["".join(sorted(s[j:j+i])) for j in range(n-i+1)]
        b = Counter(a)
        count.append(sum([len(list(combinations(['a']*b[j], 2))) for j in b]))
    return sum(count)


def sherlockAndAnagrams2(s: str) -> int:
    count = 0
    for i in range(1,len(s)+1):
        a = ["".join(sorted(s[j:j+i])) for j in range(len(s)-i+1)]
        b = Counter(a)
        for j in b:
            count += b[j] * (b[j]-1) / 2
    return int(count)


def sherlockAndAnagrams3(s: str) -> int:
    # frequency dict of sorted substrings
    s_comb_count = Counter(''.join(sorted(s[i:j])) for i, j in combinations(range(1 + len(s)), 2))
    return sum(s_comb_count[key] * (s_comb_count[key] - 1) // 2 for key in s_comb_count)


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(input())

    for q_itr in range(q):
        s = input()
        result = sherlockAndAnagrams(s)
        fptr.write(str(result) + '\n')
    fptr.close()


def test():
    data = TESTS
    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = sherlockAndAnagrams(input)
        t1 = time() - t0
        if not rst == output:
            print(f"{input} Expected: {output} Result: {rst}")
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
