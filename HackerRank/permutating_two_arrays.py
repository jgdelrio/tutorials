"""
Practice > Algorithms > Greedy > Permutating Two Arrays

Consider two n-element arrays of integers, A = [A[0], A[1],...,A[n-1] and B = [B[0], B[1],..,B[n-1]].
You want to permute them into some A' and B' such that the relation A'[i] + B'[i] >= k holds
for all i where 0<=i<n.
For example, if A=[0,1], B=[0,2], and k=1, a valid A', B' satisfying our relation would
be A'=[1,0] and B'=[0,2]

You are given q queries consisting of A, B, and k. For each query, print YES on a new line
if some permutation A', B' satisfying the relation above exists. Otherwise, print NO.

Function Description
--------------------
Complete the twoArrays function in the editor below. It should return a string, either YES or NO.

twoArrays has the following parameter(s):

k: an integer
A: an array of integers
B: an array of integers

"""

import os
from time import time

TESTS = [[(10, [2, 1, 3], [7, 8, 9]), 'YES'],
         [(5, [1, 2, 2, 1], [3, 3, 3, 4]), 'NO'],
         ]


def twoArrays2(k, A, B):
    A.sort()
    B.sort(reverse=True)
    if any((a+b < k for a, b in zip(A, B))):
        return "NO"
    else:
        return "YES"


def twoArrays(k, A, B):
    return "YES" if all(sum(c) >= k for c in zip(sorted(A), sorted(B, reverse=True))) else "NO"


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        nk = input().split()
        n = int(nk[0])
        k = int(nk[1])
        A = list(map(int, input().rstrip().split()))
        B = list(map(int, input().rstrip().split()))
        result = twoArrays(k, A, B)
        fptr.write(result + '\n')

    fptr.close()


def test():
    data = TESTS
    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = twoArrays(*input)
        t1 = time() - t0
        if not rst == output:
            print(f"{input} Expected: {output} Result: {rst}")
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
