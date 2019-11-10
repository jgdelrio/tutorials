"""
Practice > Data Structures > Advanced > Subsequence Weighting

A subsequence of a sequence is a sequence which is obtained by deleting zero or more elements from the sequence.
You are given a sequence A in which every element is a pair of integers  i.e  A = [(a1, w1), (a2, w2),..., (aN, wN)].
For a subseqence B = [(b1, v1), (b2, v2), ...., (bM, vM)] of the given sequence :
- We call it increasing if for every i (1 <= i < M ) , bi < bi+1.
- Weight(B) = v1 + v2 + ... + vM.

Task:
Given a sequence, output the maximum weight formed by an increasing subsequence.

Input:
The first line of input contains a single integer T. T test-cases follow.
The first line of each test-case contains an integer N.
The next line contains a1, a2 ,... , aN separated by a single space.
The next line contains w1, w2, ..., wN separated by a single space.

Output:
For each test-case output a single integer: The maximum weight of increasing subsequences of the given sequence.

Constraints:
1 <= T <= 5
1 <= N <= 150000
1 <= ai <= 109, where i ∈ [1..N]
1 <= wi <= 109, where i ∈ [1..N]
"""

import os
from time import time

TEST1 = ([1, 2, 3, 4], [10, 20, 30, 40], 100)
TEST2 = ([1, 2, 3, 4, 1, 2, 3, 4], [10, 20, 30, 40, 15, 15, 15, 50], 110)
TEST3 = ([1, 2, 3, 4, 2, 3, 4, 1, 2], [1, 1, 1, 1, 5, 1, 1, 9, 8], 17)

TESTS = [TEST1, TEST2, TEST3]


def solve(array, weights):
    subsequence = [0]
    subseq_w = [0]
    for idx, weight in enumerate(weights):
        if array[idx] > subsequence[-1]:
            subsequence.append(array[idx])
            subseq_w.append(weight)
        elif weight > sum(subseq_w):
            subsequence = [array[idx]]
            subseq_w = [weight]
        else:
            new_subseq = [k for k in subsequence if k < array[idx]]
            if sum(subseq_w[:len(new_subseq)]) + weight > sum(subseq_w):
                subsequence = new_subseq
                subseq_w = [*subseq_w[:len(new_subseq)], weight]

    return sum(subseq_w)


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        n = int(input())
        a = list(map(int, input().rstrip().split()))
        w = list(map(int, input().rstrip().split()))
        result = solve(a, w)
        fptr.write(str(result) + '\n')
    fptr.close()


def test():
    repeat = int(1e5)
    tt = []
    for current_test in TESTS:
        a, w, output = current_test

        # for r in range(repeat):
        t0 = time()
        result = solve(a, w)
        tt.append(time() - t0)
        print(f"Pass: {result==output}\tResult: {result}\tExpected: {output}")
    print(f'Av time: {sum(tt) / len(TESTS)}')
    print(f'Total time: {sum(tt)}')


if __name__ == '__main__':
    test()
