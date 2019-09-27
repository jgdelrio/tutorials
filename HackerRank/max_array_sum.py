"""
Max Array Sum

Given an array of integers, find the subset of non-adjacent elements with the maximum sum.
Calculate the sum of that subset.  For example, given an array   arr = [-2, 1, 3, -4, 5]
we have the  following possible subsets:

Subset        Sum
[-2, 3, 5]    6
[-2, 3]       1
[-2, -4]      -6
[-2, 5]       3
[1, -4]       -3
[1, 5]        6
[3, 5]        8
"""

import math
import os
import sys
from time import time

TEST = ([(3, 7, 4, 6, 5), 13],
        [(2, 1, 5, 8, 4), 11],
        [(3, 5, -7, 8, 10), 15])


# Complete the maxSubsetSum function below.
def maxSubsetSum(arr):
    """Accumulate for non-consecutive values"""
    current = max(arr[0], arr[1], 0)
    prev = max(arr[0], 0)
    for i in range(2, len(arr)):
        current, prev = max(current, prev + arr[i]), current

    return current


def run_hackerrun():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    res = maxSubsetSum(arr)
    fptr.write(str(res) + '\n')
    fptr.close()


def test():
    data = TEST
    repeat = int(1)
    t = []
    for r in range(repeat):
        for d in data:
            data_in, data_exp = d
            t0 = time()
            data_out = maxSubsetSum(data_in)
            t.append(time() - t0)
            assert data_out == data_exp
    print(f'Av time: {sum(t) / repeat}')
    print(f'Total time: {sum(t)}')


if __name__ == '__main__':
    test()

