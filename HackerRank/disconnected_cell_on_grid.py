"""
Practice > Tutorials > Cracking the Coding Interview > DFS: Connected Cell in a Grid

Consider a matrix where each cell contains either a 0 or a 1 and any cell containing a 1 is
called a filled cell. Two cells are said to be connected if they are adjacent to each other
horizontally, vertically, or diagonally.

If one or more filled cells are also connected (vertically, horizontally or diagonally), they
form a region. Note that each cell in a region is connected to at least one other cell in the
region but is not necessarily directly connected to all the other cells in the region.

Given an   n x m   matrix, find and print the number of cells in the largest region in the matrix.
"""

import os
from time import time

TEST1 = ([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]], 5)


def maxRegion(grid):
    # grid = np.array([np.array(k) for k in grid])
    r = len(grid)
    c = len(grid[0])

    def get_group_size(i, j):
        # For every position we recursively check the size
        if (i in range(r)) and (j in range(c)):
            if grid[i][j] == 1:
                # When a value is found we make it 0 to avoid counting it twice (destructive strategy)
                grid[i][j] = 0
                # Then we send the check for all range from current position to surroundings
                return 1 + sum(get_group_size(i2, j2) for i2 in range(i - 1, i + 2) for j2 in range(j-1, j+2))
        return 0

    return max(get_group_size(i, j) for i in range(r) for j in range(c))


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    m = int(input())
    grid = []

    for _ in range(n):
        grid.append(list(map(int, input().rstrip().split())))
    res = maxRegion(grid)
    fptr.write(str(res) + '\n')
    fptr.close()


def test():
    data = [TEST1]

    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = maxRegion(input)
        t1 = time() - t0
        assert (rst == output)
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
