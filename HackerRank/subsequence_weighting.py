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

TESTS = [TEST2, TEST1, TEST3]


def solve0(array, weights):
    sequence = [(0, 0)]
    for ref, weight in zip(array, weights):
        if ref > sequence[-1][0]:
            # We can add the element to the subsequence
            sequence.append((ref, sequence[-1][1] + weight))
        elif weight > sequence[-1][1]:
            # The previous subsequence doesn't matter as this weight is already bigger
            sequence = [(ref, weight)]
        else:
            # We must select if to keep part of the previous subseq or make a new one
            idx = 0
            while sequence[idx][0] < ref:
                idx += 1
            new_accum_weight = sequence[idx-1][1] + weight
            if new_accum_weight > sequence[-1][1]:
                sequence = sequence[:idx]
                sequence.append((ref, new_accum_weight))
    return sequence[-1][1]


def solve(array, weights):
    values, accu = [0], [0]
    for ref, weight in zip(array, weights):
        if ref > values[-1]:
            # We can add the element to the subsequence
            values.append(ref)
            accu.append(accu[-1] + weight)
        elif weight > accu[-1]:
            # The previous subsequence doesn't matter as this weight is already bigger
            values, accu = [ref], [weight]
        else:
            # We must select if to keep part of the previous subseq or make a new one
            try:
                idx = values.index(ref)
            except:
                idx = 0
                while values[idx] < ref:
                    idx += 1
            new_accum = accu[idx-1] + weight
            if new_accum > accu[-1]:
                values, accu = values[:idx], accu[:idx]
                values.append(ref)
                accu.append(new_accum)
    return accu[-1]


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
    repeat = int(1e3)
    # 0.00317  0.0095
    # 0.0027   0.0082
    # 0.0021   0.0065
    tt = []
    for r in range(repeat):
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
