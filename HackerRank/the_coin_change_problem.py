"""
Practice > Algorithms > Dynamic Programming > The Coin Change Problem

You are working at the cash counter at a fun-fair, and you have different types
of coins available to you in infinite quantities. The value of each coin is
already given. Can you determine the number of ways of making change for a
particular number of units using the given types of coins?

For example, if you have 4 types of coins, and the value of each type
is given as 8,3,1,2 respectively, you can make change for 3 units in three ways:
   {1,1,1}, {1,2} and {3}.


Function Description
--------------------
Complete the getWays function in the editor below. It must return an integer
denoting the number of ways to make change.

getWays has the following parameter(s):

n: an integer, the amount to make change for
c: an array of integers representing available denominations

"""

import os
from time import time


TESTS = [
    [[10, [2, 5, 3, 6]], 5],
]


memory_dict = {}


def getWays(n: int, c: int) -> int:
    global memory_dict
    clen = len(c)
    if n < 0:
        return 0
    elif n == 0:
        return 1
    elif (n, clen) in memory_dict.keys():
        return memory_dict[(n, clen)]
    else:
        cum_sum = 0
        for c_index in range(clen):
            c_item = c[c_index]
            if c_item > n:
                continue
            else:
                result = getWays(n-c_item, c[c_index:])
                if result > 0:
                    cum_sum += result
        memory_dict[(n, clen)] = cum_sum
        return cum_sum


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = int(first_multiple_input[0])
    m = int(first_multiple_input[1])
    c = list(map(int, input().rstrip().split()))

    # Print the number of ways of making change for 'n' units
    # using coins having the values given by 'c'
    ways = getWays(n, c)
    fptr.write(str(ways) + '\n')
    fptr.close()


def test() -> None:
    data = TESTS
    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = getWays(*input)
        t1 = time() - t0
        if not rst == output:
            print(f"{input} Expected: {output} Result: {rst}")
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
