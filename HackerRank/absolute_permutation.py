"""
Algorithms > Implementation > Absolute Permutation

We define 'P' to be a permutation of the first 'n' natural numbers in the range [1, n].
Let 'pos[i]' denote the value at position 'i' in permutation P using 1-based indexing.
P is considered to be an absolute permutation if   |pos[i] - i| = k   holds true for every i in [1, n].

Given n and k, print the lexicographically smallest absolute permutation P.
If no absolute permutation exists, print -1.

For example, let  n = 4   giving us an array  pos = [1, 2, 3, 4]. If we use 1 based indexing,
create a permutation where every   |pos[i] - i| = k. If k = 2, we could rearrange them to [3, 4, 1, 2]:
"""
import os
from time import time

TEST1 = ([[2, 1], [2, 1]],
         [[3, 0], [1, 2, 3]],
         [[3, 2], [-1]],
         [[100, 2], [3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10, 15, 16, 13, 14, 19, 20, 17, 18, 23, 24,
                     21, 22, 27, 28, 25, 26, 31, 32, 29, 30, 35, 36, 33, 34, 39, 40, 37, 38, 43, 44,
                     41, 42, 47, 48, 45, 46, 51, 52, 49, 50, 55, 56, 53, 54, 59, 60, 57, 58, 63, 64,
                     61, 62, 67, 68, 65, 66, 71, 72, 69, 70, 75, 76, 73, 74, 79, 80, 77, 78, 83, 84,
                     81, 82, 87, 88, 85, 86, 91, 92, 89, 90, 95, 96, 93, 94, 99, 100, 97, 98]])


def absolutePermutation(n, k):
    result = []
    s = set(range(1, n + 1))       # 1-indexing notation.
    # Note: Using set instead of list solves the time out for certain cases

    for pos in s.copy():
        val = pos - k
        if val in s:
            # find the s number that will allow to comply with the requirement
            result.append(val)
            s.remove(val)       # once used extract the s from the list of available candidates
        elif pos + k in s:          # try the same thing with the sum
            result.append(pos + k)
            s.remove(pos + k)
        else:
            return [-1]

    return result


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())

    for t_itr in range(t):
        nk = input().split()
        n = int(nk[0])
        k = int(nk[1])
        result = absolutePermutation(n, k)
        fptr.write(' '.join(map(str, result)))
        fptr.write('\n')

    fptr.close()


def test():
    data = TEST1
    repeat = int(1e5)
    t = []

    for r in range(repeat):
        for d in data:
            input = d[0]
            output = d[1]

            t0 = time()
            rst = absolutePermutation(input[0], input[1])
            t.append(time() - t0)
            assert (rst == output)
    print(f'Av time: {sum(t) / repeat}')
    print(f'Total time: {sum(t)}')


if __name__ == '__main__':
    test()
