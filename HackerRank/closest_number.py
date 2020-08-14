"""
Practice > Mathematics > Number Theory > Closest Number


"""

import os
from time import time

TESTS = [((349, 1, 4), 348),
         ((395, 1, 7), 392),
         ((4, -2, 2), 0)]


def findPoint(px, py, qx, qy):
    return 2 * qx - px, 2 * qy - py


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    for n_itr in range(n):
        pxPyQxQy = input().split()
        px = int(pxPyQxQy[0])
        py = int(pxPyQxQy[1])
        qx = int(pxPyQxQy[2])
        qy = int(pxPyQxQy[3])

        result = findPoint(px, py, qx, qy)

        fptr.write(' '.join(map(str, result)))
        fptr.write('\n')

    fptr.close()


def test():
    data = TESTS
    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = findPoint(*input)
        t1 = time() - t0
        if not rst == output:
            print(f"{input} Expected: {output} Result: {rst}")
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
