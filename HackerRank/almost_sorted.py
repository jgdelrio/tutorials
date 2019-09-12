"""
Algorithms > Implementation > Almost Sorted

Given an array of integers, determine whether the array can be sorted in ascending order
using only one of the following operations one time:
  1. Swap two elements.
  2. Reverse one sub-segment.

Determine whether one, both or neither of the operations will complete the task.
If both work, choose swap. For instance, given an array [2, 3, 5, 4] either swap the 4 and 5,
or reverse them to sort the array. Choose swap. The Output Format section below details requirements.

Function Description:
Complete the almostSorted function in the editor below. It should print the results and return nothing.
almostSorted has the following parameter(s):
    arr: an array of integers

Input Format:
    The first line contains a single integer 'n', the size of 'arr'.
    The next line contains n space-separated integers 'arr[i]' where  1 <= i <= n.

Constraints:
    2 <= n <= 100000
    0 <= arr[i] <= 1e6
All arr[i] are distinct

Output Format:
  1. If the array is already sorted, output yes on the first line. You do not need to output anything else.
  2. If you can sort this array using one single operation (from the two permitted operations)
  then output yes on the first line and then:
    a) If elements can be swapped, d[l] and d[r], output swap 'l' 'r' in the second line. 'l' and 'r' are
    the indices of the elements to be swapped, assuming that the array is indexed from 1 to n.
    b) Otherwise, when reversing the segment 'd[l...r]', output reverse 'l' 'r' in the second line.
     'l' and 'r' are the indices of the first and last elements of the subsequence to be reversed,
     assuming that the array is indexed from 1 to n.
     If and array can be sorted by either swapping or reversing, choose swap.
  3. If you cannot sort the array either way, output no on the first line.
"""
from time import time

TEST1 = ([4, 2], 'yes swap 1 2')
TEST2 = ([3, 1, 2], 'no')
TEST3 = ([1, 5, 4, 3, 2, 6], 'yes reverse 2 5')


def almostSorted(arr):
    idx_lower = 0
    idx_upper = len(arr) - 1
    
    # move up the index while the list is sorted
    while idx_lower < idx_upper and arr[idx_lower] < arr[idx_lower + 1]:
        idx_lower += 1
    
    # if we reach the upper index means that the full list is already sorted
    if idx_lower == idx_upper:
        return 'yes'
    
    # as with the lower index we move down the upper index while we still have a sorted list
    while idx_upper > 0 and arr[idx_upper] > arr[idx_upper - 1]:
        idx_upper -= 1
    
    # Now we find if either of the two operations sort the list
    if idx_lower == 0 or arr[idx_upper] > arr[idx_lower - 1]:
        # we can swap
        if arr[idx_upper] < arr[idx_lower + 1] or idx_lower + 1 == idx_upper:
            # high index swapable
            if idx_upper == len(arr) - 1 or arr[idx_lower] < arr[idx_upper + 1]:
                # low index swapable
                if arr[idx_lower] > arr[idx_upper - 1] or idx_lower == idx_upper - 1:
                    idx_lower_run = idx_lower + 1
                    while idx_lower_run < idx_upper and arr[idx_lower_run] < arr[idx_lower_run + 1]:
                        idx_lower_run += 1
                    if idx_lower_run == idx_upper - 1 or idx_lower == idx_upper - 1:
                        return f'yes swap {idx_lower + 1} {idx_upper + 1}'

    idx_lower_run = idx_lower + 1
    while idx_lower_run < idx_upper and arr[idx_lower_run] > arr[idx_lower_run + 1]:
        idx_lower_run += 1

    if idx_lower_run == idx_upper:
        if idx_lower == 0 or arr[idx_upper] > arr[idx_lower - 1]:
            if idx_upper == len(arr) - 1 or arr[idx_lower] < arr[idx_upper + 1]:
                return f'yes reverse {idx_lower + 1} {idx_upper + 1}'

    return 'no'


def hackerrank_run():
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    almostSorted(arr)


def test():
    data = [TEST1, TEST2]
    repeat = int(1e5)
    t = []

    for r in range(repeat):
        for d in data:
            input = d[0]
            output = d[1]

            t0 = time()
            rst = almostSorted(input)
            t.append(time() - t0)
            assert (rst == output)
    print(f'Av time: {sum(t) / repeat}')
    print(f'Total time: {sum(t)}')


if __name__ == '__main__':
    test()






