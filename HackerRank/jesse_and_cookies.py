"""
Data Structures > Heap

Jesse loves cookies. He wants the sweetness of all his cookies to be greater than value 'K'.
To do this, Jesse repeatedly mixes two cookies with the least sweetness. He creates a special combined cookie with:

sweetness = (1 x Least sweet cookie + 2 X 2nd least sweet cookie)

He repeats this procedure until all the cookies in his collection have a sweetness >= K.
You are given Jesse's cookies. Print the number of operations required to give the cookies a sweetness >= K.
Print -1 if this isn't possible.

Input Format:
The first line consists of integers 'N', the number of cookies and 'K',
the minimum required sweetness, separated by a space.  The next line contains 'N' integers describing the
array 'A' where 'Ai' is the sweetness of the ith cookie in Jesse's collection.

Constraints:
1 <= N <= 10^6
1 <= K <= 10^9
1 <= Ai <= 10^6

Output Format:
Output the number of operations that are needed to increase the cookie's sweetness >= K.
Output -1 if this isn't possible.
"""

import os

# Define input and output
TEST1 = ((7, [1, 2, 3, 9, 10, 12]), 2)
TEST2 = ((7, [3, 9, 10, 12, 1, 2]), 2)
TEST3 = ((7, [1, 1, 1, 1, 1, 1, 1, 1]), 6)
TEST4 = ((1, [1, 1, 1, 1, 1, 1, 1, 1]), 0)      # case where the min is already achieved without operations


def cookies_naive(k, cookie_list):
    cookie_list.sort()
    operations = 0

    while min(cookie_list) < k:
        cookie_list.sort()
        if len(cookie_list) < 2:
            return -1
        else:
            new_cookie = cookie_list[0] + 2 * cookie_list[1]
            cookie_list = [new_cookie, *cookie_list[2:]]
            operations += 1

    return operations


from heapq import heappop, heappush, heapify


def cookies(k, cookie_list):
    """
    To avoid timing-out in the test, we require a final complexity of  O(nlogn + number of operations)
    For that we use a min heap priority queue

    :param k:            sweetness desired
    :param cookie_list:  list of cookies
    :return:             number of operations
    """
    heapify(cookie_list)
    operations = 0
    n_cookies = len(cookie_list)

    while cookie_list[0] < k:
        if n_cookies < 2:
            return -1
        else:
            cookie1 = heappop(cookie_list)
            cookie2 = heappop(cookie_list)
            new_cookie = cookie1 + 2 * cookie2
            heappush(cookie_list, new_cookie)
            operations += 1
            n_cookies -= 1

    return operations


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = int(nk[0])
    k = int(nk[1])
    cookie_list = list(map(int, input().rstrip().split()))

    result = cookies(k, cookie_list)
    fptr.write(str(result) + '\n')
    fptr.close()


def test():
    data = TEST4
    k = data[0][0]
    cookie_list = data[0][1]
    output = data[1]

    result = cookies(k, cookie_list)
    print("Operations: {}".format(result))
    assert(result == output)


if __name__ == '__main__':
    test()
