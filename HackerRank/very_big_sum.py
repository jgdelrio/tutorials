"""
Algorithms > Warmup > A Very Big Sum

Calculate and print the sum of the elements in an array,
keeping in mind that some of those integers may be quite large.

"""
import os
from time import time


TEST1 = ([1000000001, 1000000002, 1000000003, 1000000004, 1000000005], 5000000015)


def aVeryBigSum(ar):
    return sum(ar)


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    ar_count = int(input())
    ar = list(map(int, input().rstrip().split()))
    result = aVeryBigSum(ar)
    fptr.write(str(result) + '\n')
    fptr.close()


def test():
    data = TEST1
    input = data[0]
    output = data[1]

    t0 = time()
    rst = aVeryBigSum(input)
    print("Total time: {}".format(round(time() - t0, 8)))
    print("Final output: {}\nExpected: {}".format(rst, output))
    assert (rst == output)


if __name__ == '__main__':
    test()