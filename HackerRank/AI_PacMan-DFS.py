"""
Dashboard  > Artificial Intelligence > A* Search > PacMan - DFS

AI has a lot of problems that involve searches.
In this track you will learn most of the search techniques used in AI.

In this game, PacMan is positioned in a grid.
PacMan has to find the food using Depth First Search (DFS).
Assume the grid is completely observable, perform a DFS on the grid
and then print the path obtained by DFS from the PacMan to the food.

Input Format
------------
The first line contains 2 space separated integers which is the position of the PacMan.
The second line contains 2 space separated integers which is the position of the food.
The third line of the input contains 2 space separated integers indicating the size of the rows and columns respectively. The largest grid size is 30x30.

This is followed by row (r) lines each containing column (c) characters. A wall is represented by the character '%' ( ascii value 37 ), PacMan is represented by UpperCase alphabet 'P' ( ascii value 80 ), empty spaces which can be used by PacMan for movement is represented by the character '-' ( ascii value 45 ) and food is represented by the character '.' ( ascii value 46 )

You have to mark the nodes explored while populating it into the stack and not when its expanded.


Note
+ The grid is indexed as per matrix convention
+ The evaluation process follows iterative-DFS and not recursive-DFS.

Populating Stack

In order to maintain uniformity across submissions, please follow the below mentioned order in pushing nodes to stack. If a node has all the 4 adjacent neighbors. Then,

UP is inserted first into the stack, followed by LEFT, followed by RIGHT and then by DOWN.

so, if (1,1) has all its neighbors not visited, (0,1), (1,0), (1,2), (2,1) then,

(0,1) - UP is inserted first
(1,0) - LEFT is inserted second
(1,2) - RIGHT is inserted third
(2,1) - DOWN is inserted fourth (on top)
So, (2,1) is the first to be popped from the stack.

Constraints

1 <= r,c <= 40

Output Format

Each cell in the grid is represented by its position in the grid (r,c). PacMan can move only UP, DOWN, LEFT or RIGHT. Your task is to print all the nodes that you encounter while printing DFS tree. While populating the stack, the following convention must be followed.
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
