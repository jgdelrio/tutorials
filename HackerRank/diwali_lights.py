"""
Mathematics > Fundamentals > Diwali Lights

On the eve of Diwali Hari is decorating his house with a serial light bulb set.
The serial light bulb set has N bulbs placed sequentially on a string which is programmed to
change patterns every second. If at least one bulb in the set is on at any given instant of
time, how many different patterns of light can the serial light bulb set produce?

Note: Lighting two bulbs *-* is different from **-

Comment: Unless the module is calculated with the pow function, doing it after pow takes too much time
         and the tests will fail because of timeout
"""

import os

TEST1 = ([1, 2, 3], [1, 3, 7])
TEST2 = ([4], [15])


def pow_mod(x, y, z):
    """Another way to calculate (x ** y) % z efficiently"""
    number = 1
    while y:
        if y & 1:
            number = number * x % z
        y >>= 1
        x = x * x % z
    return number


def lights(n):
    p = (int)(1e5)
    return pow(2, n, p) - 1


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        n = int(input())

        result = lights(n)

        fptr.write(str(result) + '\n')

    fptr.close()


def test():
    data = TEST2
    input = data[0]
    output = data[1]

    rst = []
    for i in input:
        rst.append(lights(i))
    print("FInal output: {}".format(rst))
    assert (rst == output)


if __name__ == '__main__':
    test()
