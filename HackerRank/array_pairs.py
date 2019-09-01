"""
Data Structures > Trees > Array Pairs

Consider an array of 'n' integers, A = [a1, a2, ..., an].
Find and print the total number of (i,j) pairs such that ai x aj <= max(ai, ai+1,..., aj) where i < j.

Input Format:
The first line contains an integer, n, denoting the number of elements in the array.
The second line consists of 'n' space-separated integers describing the respective values of a1, a2, ..., an.

Constraints:
1 <= n <= 5 x 1e5
1 <= ai <= 1e9
"""

import os
import sys
from time import time

sys.setrecurionlimit(10000)

TEST1 = ([1, 1, 2, 4, 2], 8)
TEST2 = ([1, 1, 2, 4, 1], 9)
TEST3 = ([1, 1, 4, 1, 1], 10)
TEST4 = ([1, 1, 4, 2, 1, 1, 8, 1, 1, 2, 8, 4], 55)
TEST5 = ([1, 8, 4, 2, 1, 3, 5, 12, 1, 12, 8, 1, 1], 50)
TEST6 = ([1, 1, 4, 2, 1, 1, 8, 1, 1, 2, 8, 4, 6, 12, 1, 5, 7, 9, 1, 3, 1, 10], 163)
TEST7 = ([*[1] * 100, 10, *[1] * 500, 20, *[1] * 300], 406350)
TEST8 = ([*[*[1] * 1000, 10, *[1] * 500, 20, *[1] * 1000] * 3], 28166250)
TEST9 = ([1, 1, 2, 3, 1, 2], 12)


def naive_solve(arr):
    """Validation function"""
    counter = 0
    for n, i in enumerate(arr):
        for m, j in enumerate(arr[n+1:]):
            max_val = max(arr[n:(n+m+2)])
            if i * j <= max_val:
                counter += 1

    return counter


def solve(arr):
    counter = 0
    m = max(arr)
    idx = arr.index(m)
    left = arr[:idx]
    right = arr[idx+1:]
    if len(left) > 1:
        counter += solve(left)
    if len(right) > 1:
        counter += solve(right)

    counter += left.count(1)
    counter += right.count(1)

    left.sort()
    right.sort()
    if len(right) > 0:
        for l in left:
            if l * right[0] <= m:
                counter += 1
            else:
                break

            for r in right[1:]:
                if l * r <= m:
                    counter += 1
                else:
                    break
    return counter


def solve_just1s(arr, m=None):
    counter = 0

    # Get max and divide list
    m = max(arr)
    idx = arr.index(m)
    left = arr[:idx]
    right = arr[idx+1:]

    n_left = len(left)
    n_right = len(right)

    # Count 1's and update counter
    if n_left == 0:
        counter += right.count(1)           # pairs with the max
        if n_right == 2:
            counter += (1 in right)
        elif n_right > 1:
            counter += solve_just1s(right)      # pairs within
        return counter

    elif n_right == 0:
        counter += left.count(1)
        if n_left == 2:
            counter += (1 in left)
        elif n_left > 1:
            counter += solve_just1s(left)
        return counter

    else:
        left_ones = left.count(1)
        right_ones = right.count(1)
        counter += left_ones + right_ones

    if n_right > 1:
        counter += solve_just1s(right)
    if n_left > 1:
        counter += solve_just1s(left)

    left.sort()
    right.sort()
    if len(right) > 0:
        for l in left:
            if l * right[0] <= m:
                counter += 1
            else:
                break

            for r in right[1:]:
                if l * r <= m:
                    counter += 1
                else:
                    break

    # if n_left > 0 and n_right > 0:
    #     counter += (left_ones * sum([k<=m for k in right]))
    #     counter += (right_ones * sum([(k>1 and k<=m) for k in left]))

    return counter


def solve_extra(arr):
    m = max(arr)
    if m == 1:
        list_len = len(arr)
        return list_len * (list_len - 1) / 2

    counter = 0
    idx = arr.index(m)
    left = arr[:idx]
    right = arr[idx + 1:]

    len_left = len(left)
    len_right = len(right)

    if len_left > 1:
        counter += solve_extra(left)
    if len_right > 1:
        counter += solve_extra(right)

    if len_left > 0:
        counter += left.count(1)
    if len_right > 0:
        counter += right.count(1)

    if len_right > 0 and len_left > 0:
        left.sort(reverse=True)
        right.sort(reverse=True)

        for r_ind, er in enumerate(right):
            for l_ind, el in enumerate(left):
                if er * el <= m:
                    if l_ind == 0:
                        counter += ((len_right - r_ind) * len_left)
                        return counter
                    else:
                        counter += len_left - l_ind
                        break
    return counter


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    arr_count = int(input())
    arr = list(map(int, input().rstrip().split()))
    result = int(solve(arr))
    fptr.write(str(result) + '\n')
    fptr.close()


def test():
    datas = [TEST1, TEST2, TEST3, TEST4, TEST5, TEST6, TEST7, TEST9, TEST8]
    # datas = [TEST4]

    for data in datas:
        input = data[0]
        output = data[1]

        t0 = time()
        rst = int(solve_extra(input))
        print("Total time: {}".format(round(time() - t0, 8)))
        print("Final output: {}\nExpected: {}".format(rst, output))
        assert (rst == output)


if __name__ == '__main__':
    test()
