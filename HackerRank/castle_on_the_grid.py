"""
Data Structures > Queues > Castle on the Grid

You are given a square grid with some cells open (.) and some blocked (X).
Your playing piece can move along any row or column until it reaches the edge of the grid
or a blocked cell. Given a grid, a start and an end position, determine the number of moves
it will take to get to the end position.

For example, you are given a grid with sides   n = 3   described as follows:

   ...
   .X.
   ...

Your starting position   (startX, startY) = (0, 0)   so you start in the top left corner.
The ending position is  (goalX, goalY) = (1, 2)    The path is  (0,0) -> (0,2) -> (1,2).
It takes 2 moves to get to the goal.

Complete the minimumMoves function in the editor. It must print an integer denoting the minimum
moves required to get from the starting position to the goal.

"""

import os
from time import time
from collections import deque

TEST1 = ([['.X.', '.X.', '...'], 0, 0, 0, 2], 3)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "X=%d,Y=%d" % (self.x, self.y)


def getPointsFromPoint(N, arr, point):
    x = point.x
    y = point.y
    points = []

    while x > 0:
        x -= 1
        if arr[x][y] == 'X':
            break
        points.append(Point(x, y))

    x = point.x
    while x < N - 1:
        x += 1
        if arr[x][y] == 'X':
            break
        points.append(Point(x, y))

    x = point.x
    while y > 0:
        y -= 1
        if arr[x][y] == 'X':
            break
        points.append(Point(x, y))

    y = point.y
    while y < N - 1:
        y += 1
        if arr[x][y] == 'X':
            break
        points.append(Point(x, y))

    return points


def solveCastleGrid(N, arr, start, end):
    q = deque([start])          # double-ended queue
    arr[start.x][start.y] = 0

    while q:
        current_point = q.pop()
        current_distance = arr[current_point.x][current_point.y]

        points = getPointsFromPoint(N, arr, current_point)
        for p in points:
            if arr[p.x][p.y] == '.':
                arr[p.x][p.y] = current_distance + 1
                q.appendleft(p)
                if p.x == end.x and p.y == end.y:
                    return current_distance + 1
    return -1


def minimumMoves(grid, startX, startY, goalX, goalY):
    return solveCastleGrid(len(grid), list(map(list, grid)), Point(startX, startY), Point(goalX, goalY))


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    grid = []

    for _ in range(n):
        grid_item = input()
        grid.append(grid_item)

    startXStartY = input().split()
    startX = int(startXStartY[0])
    startY = int(startXStartY[1])
    goalX = int(startXStartY[2])
    goalY = int(startXStartY[3])
    result = minimumMoves(grid, startX, startY, goalX, goalY)
    fptr.write(str(result) + '\n')
    fptr.close()


def test():
    data = TEST1

    input = data[0]
    output = data[1]

    grid, startX, startY, goalX, goalY = input

    t0 = time()
    rst = minimumMoves(grid, startX, startY, goalX, goalY)
    t1 = time() - t0
    assert (rst == output)
    print(f'Total time: {t1}')


if __name__ == '__main__':
    test()



