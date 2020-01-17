"""
Maximum Element
  1- Push the element x into the stack.
  2- Delete the element present at the top of the stack.
  3- Print the maximum element in the stack.
"""

import os
from time import time

TEST = ([[1, 97], [2], [1, 20], [2], [1, 26], [1, 20], [2], [3], [1, 91], [3]],
        [26, 91])


def maximum_element(arr):
    stack = []
    rst = []
    for elem in arr:
        if elem[0] == 1:
            # Add element
            if len(stack) > 0:
                stack.append([elem[1], max(stack[-1][1], elem[1])])
            else:
                stack.append([elem[1], elem[1]])
        elif elem[0] == 2:
            # Delete top element
            stack.pop()
        elif elem[0] == 3:
            # Append value to print result
            rst.append(stack[-1][1])
    return rst


def run_hackerrun():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    arr = []
    for k in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    rst = maximum_element(arr)
    fptr.write("\n".join(list(map(str, rst))))
    fptr.close()


def test():
    data = TEST
    repeat = int(1)
    t = []
    for r in range(repeat):
        data_in, data_exp = data
        t0 = time()
        data_out = maximum_element(data_in)
        t.append(time() - t0)
        assert data_out == data_exp
    print(f'Av time: {sum(t) / repeat}')
    print(f'Total time: {sum(t)}')


if __name__ == '__main__':
    test()

