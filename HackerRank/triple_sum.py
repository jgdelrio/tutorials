"""
Given 3 arrays a,b,c of different sizes, find the number of distinct triplets (p,q,r)
where p is an element of a, q of b and r of c and satisfying the criteria: p<=q and q>=r.

For example, given a = [3,5,7], b=[3,6] and c=[4,6,9] we find four distinct triplets:
(3,6,4), (3,6,6), (5,6,4), (5,6,6)

Function Description
Complete the triplets function. It must return the number of distinct triplets
that can be formed from the given arrays.

triplets has the following parameter(s):
   a, b, c: three arrays of integers .

Input Format
The first line contains 3 integers lena, lenb, lenc, the sizes of the three arrays.
The next 3 lines contain space-separated integers numbering lena, lenb, lenc respectively.

Constraints 
1 <= lena, lenb, lenc <= 10e5
1 <= a,b,c <= 10e8

Output Format
Print an integer representing the number of distinct triplets.
"""

import os
from time import time

TEST1 = ([1, 3, 5], [2, 3], [1, 2, 3], 8)
TEST2 = ([1, 4, 5], [2, 3, 3], [1, 2, 3], 5)
TEST3 = ([1, 3, 5, 7], [5, 7, 9], [7, 9, 11, 13], 12)

TESTS = [TEST1, TEST2, TEST3]


def triplets(a, b, c):
    # 1- They ask for distinct triplets, then clean repetitions
    a = list(sorted(set(a)))
    b = list(sorted(set(b)))
    c = list(sorted(set(c)))

    # Initialization
    ai, bi, ci = 0, 0, 0
    counter = 0

    # Start running throughout b
    while bi < len(b):
        while ai < len(a) and a[ai] <= b[bi]:
            # move ai counter to include all 'a' <= current 'b'
            ai += 1
        while ci < len(c) and c[ci] <= b[bi]:
            # move ci counter to include all 'c' <= current 'b'
            ci += 1
        # Combinations is just the multiplication of number of 'a' and number of 'c'
        counter += ai * ci
        # Move to next 'b'
        bi += 1
    return counter


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    lenaLenbLenc = input().split()
    lena = int(lenaLenbLenc[0])
    lenb = int(lenaLenbLenc[1])
    lenc = int(lenaLenbLenc[2])
    arra = list(map(int, input().rstrip().split()))
    arrb = list(map(int, input().rstrip().split()))
    arrc = list(map(int, input().rstrip().split()))
    ans = triplets(arra, arrb, arrc)
    fptr.write(str(ans) + '\n')
    fptr.close()


def test():
    repeat = int(1e5)
    tt = []
    for t in TESTS:
        a, b, c, output = t

        # for r in range(repeat):
        t0 = time()
        result = triplets(a, b, c)
        tt.append(time() - t0)
        print(f"Pass: {result==output}\tResult: {result}\tExpected: {output}")
    print(f'Av time: {sum(tt) / len(TESTS)}')
    print(f'Total time: {sum(tt)}')


if __name__ == '__main__':
    test()
