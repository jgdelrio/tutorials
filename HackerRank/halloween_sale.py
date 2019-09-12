"""
Algorithms Implementation Halloween Sale

You wish to buy video games from the famous online video game store Mist.
Usually, all games are sold at the same price, 'p' dollars. However, they are planning to
have the seasonal Halloween Sale next month in which you can buy games at a cheaper price.
Specifically, the first game you buy during the sale will be sold at dollars, but every
subsequent game you buy will be sold at exactly 'p' dollars less than the cost of the previous
one you bought. This will continue until the cost becomes less than or equal to 'm' dollars,
after which every game you buy will cost 'm' dollars each.

For example if p = 20, d = 3 and m = 6 then the following are the costs of the first 11 games you buy, in order:
   [20, 17, 14, 11, 8, 6, 6, 6, 6, 6, 6]
"""

import os
from math import floor, sqrt

TEST1 = ([20, 3, 6, 85], 7)


def howManyGames(p, d, m, s):
    n = floor((p - m) / d + 1)
    if ((n * (2 * p - (n - 1) * d)) / 2 <= s):
        return floor(n + (s - (n * (2 * p - (n - 1) * d) / 2)) / m)
    else:
        return floor(((-d - 2 * p) + sqrt((-2 * p - d) * (-2 * p - d) - (8 * d * s))) / (-2 * d))


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    pdms = input().split()
    p = int(pdms[0])
    d = int(pdms[1])
    m = int(pdms[2])
    s = int(pdms[3])

    answer = howManyGames(p, d, m, s)
    fptr.write(str(answer) + '\n')
    fptr.close()


def test():
    data = TEST1
    input = data[0]
    output = data[1]

    rst = howManyGames(input[0], input[1], input[2], input[3])
    assert (rst == output)


if __name__ == '__main__':
    test()
