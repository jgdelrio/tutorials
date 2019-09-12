"""
Data Structures > Advanced > Jim and the Skyscrapers

Jim has invented a new flying object called HZ42. HZ42 is like a broom and can only fly
horizontally, independent of the environment. One day, Jim started his flight from Dubai's
highest skyscraper, traveled some distance and landed on another skyscraper of same height!
So much fun! But unfortunately, new skyscrapers have been built recently.

Let us describe the problem in one dimensional space. We have in total 'N' skyscrapers aligned
from left to right. The ith skyscraper has a height of hi. A flying route can be described as (i,j)
with   i != j  , which means, Jim starts his HZ42 at the top of the skyscraper i and lands on the
skyscraper j. Since HZ42 can only fly horizontally, Jim will remain at the height hi only. Thus the
path (i,j) can be valid, only if each of the skyscrapers  i,i+1,...,j-1,j   is not strictly greater
than hi and if the height of the skyscraper he starts from and arrives on have the same height.
Formally, (i,j) is valid iff  notExistk in [i,j]: h_k >h_i and  h_i = h_j.

Help Jim in counting the number of valid paths represented by ordered pairs (i,j).

Input Format:
    The first line contains N, the number of skyscrapers. The next line contains N space separated
    integers representing the heights of the skyscrapers.

Output Format:
    Print an integer that denotes the number of valid routes.

Constraints:
    1 <= N <= 3x1e5 and no skyscraper will have height greater than 1e6 and less than 1.
"""
import os
from time import time

TEST1 = ([3, 2, 1, 2, 3, 3], 8)
TEST2 = ([1, 1000, 1], 0)           # there is a bigger skyscraper between 1 and 3 so it is not valid

MAX_N = int(1e5)


def build_nge(vector, n):
    s = [0]
    a = [0] * (n + 1)

    for i in range(n):
        # eval that the vector at the current last s is lower than the current vector value
        while s and vector[s[-1]] <= vector[i]:
            a[s[-1]] = i    # it will add the current index in the previous values are lower than the current
            s.pop()
        s.append(i)

    while s:
        a[s[-1]] = -1
        s.pop()

    return a


def solve(arr):
    n = len(arr)
    b = [0] * (n + 1)

    a = build_nge(arr, n)

    for i in range(n):
        if arr[i] == arr[a[i]]:
            b[a[i]] = b[i] + 1
    return 2 * sum(b[:n])


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    arr_count = int(input())
    arr = list(map(int, input().rstrip().split()))
    result = solve(arr)
    fptr.write(str(result) + '\n')
    fptr.close()


def test():
    data = [TEST1, TEST2]

    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = solve(input)
        t1 = time() - t0
        assert (rst == output)
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
