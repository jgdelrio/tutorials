"""
Algorithms > Sorting > Closest Numbers

Sorting is useful as the first step in many different tasks. The most common task is to make
finding things easier, but there are other uses as well. In this case, it will make it easier
to determine which pair or pairs of elements have the smallest absolute difference between them.

For example, if you've got the list [5, 2, 3, 4, 1], sort it as [1, 2, 3, 4, 5] to see that several
pairs have the minimum difference of 1: [(1,2), (2,3), (3,4), (4,5)]. The return array would be
[1, 2, 2, 3, 3, 4, 4, 5].

Given a list of unsorted integers, arr, find the pair of elements that have the smallest absolute
difference between them. If there are multiple pairs, find them all.

Constraints:
    2 <= n <= 200000
    -1e7 <= arr[i] <= 1e7
    All arr[i] are unique in arr.

Output Format:
    Output the pairs of elements with the smallest difference. If there are multiple pairs, output
    all of them in ascending order, all on the same line with just a single space between each pair
    of numbers. A number may be part of two pairs when paired with its predecessor and its successor.

"""
import os
from time import time

# 30 - (-20) = 50 which is the smallest difference
TEST1 = ([-20, -3916237, -357920, -3620601, 7374819, -7330761, 30, 6246457, -6461594, 266854 ], [-20, 30])

TEST2 = ([-20, -3916237, -357920, -3620601, 7374819, -7330761, 30, 6246457, -6461594, 266854, -520, -470],
         [-520, -470, -20, 30])

TEST3 = ([5, 4, 3, 2], [2, 3, 3, 4, 4, 5])


def closestNumbers(arr):
    rst = []
    arr.sort()
    smallest_difference = float('inf')      # set smallest diff as inf
    smallest_pairs = []

    for idx in range(len(arr)-1):
        diff = arr[idx+1] - arr[idx]
        if diff < smallest_difference:
            smallest_difference = diff
            smallest_pairs = [(arr[idx], arr[idx+1])]
        elif diff == smallest_difference:
            smallest_pairs.append((arr[idx], arr[idx+1]))

    for pair in smallest_pairs:
        rst.append(pair[0])
        rst.append(pair[1])
    return rst


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    result = closestNumbers(arr)
    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')
    fptr.close()


def test():
    data = [TEST1, TEST2, TEST3]

    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = closestNumbers(input)
        t1 = time() - t0
        assert (rst == output)
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
