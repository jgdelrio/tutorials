"""
Algorithms > Implementation > Cavity Map

You are given a square map as a matrix of integer strings. Each cell of the map has a value
denoting its depth. We will call a cell of the map a cavity if and only if this cell is not
on the border of the map and each cell adjacent to it has strictly smaller depth. Two cells
are adjacent if they have a common side, or edge.

Find all the cavities on the map and replace their depths with the uppercase character X.

For example, given a matrix:
  989
  191
  111

Output:
   989
   1X1
   111

The center cell was deeper than those on its edges: [8,1,1,1].
The deep cells in the top two corners don't share an edge with the center cell.
"""

import os

TEST1 = (['1112', '1912', '1892', '1234'], ['1112', '1X12', '18X2', '1234'])


def cavityMap(grid):
    n = len(grid)
    grid = list(map(list, grid))

    for idx_tb in range(1, n - 1):
        for idx_lr in range(1, n - 1):
            if grid[idx_tb - 1][idx_lr] != 'X' and int(grid[idx_tb - 1][idx_lr]) < int(grid[idx_tb][idx_lr]) and \
                    grid[idx_tb + 1][idx_lr] != 'X' and int(grid[idx_tb + 1][idx_lr]) < int(grid[idx_tb][idx_lr]) and \
                    grid[idx_tb][idx_lr - 1] != 'X' and int(grid[idx_tb][idx_lr - 1]) < int(grid[idx_tb][idx_lr]) and \
                    grid[idx_tb][idx_lr + 1] != 'X' and int(grid[idx_tb][idx_lr + 1]) < int(grid[idx_tb][idx_lr]):
                grid[idx_tb][idx_lr] = 'X'

    return list(map(''.join, grid))


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    grid = []

    for _ in range(n):
        grid_item = input()
        grid.append(grid_item)

    result = cavityMap(grid)
    fptr.write('\n'.join(result))
    fptr.write('\n')
    fptr.close()


def test():
    data = TEST1

    input = data[0]
    output = data[1]

    rst = cavityMap(input)
    assert (rst == output)


if __name__ == '__main__':
    test()

