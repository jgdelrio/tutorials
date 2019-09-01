"""
Algorithms > Search > Count Luck

Ron and Hermione are deep in the Forbidden Forest collecting potion ingredients, and they've managed
to lose their way. The path out of the forest is blocked, so they must make their way to a portkey
that will transport them back to Hogwarts.

Consider the forest as an N x M grid. Each cell is either empty (represented by .) or blocked by a tree
(represented by X). Ron and Hermione can move (together inside a single cell) LEFT, RIGHT, UP, and DOWN
through empty cells, but they cannot travel through a tree cell. Their starting cell is marked with the
character M, and the cell with the portkey is marked with a *. The upper-left corner is indexed as (0,0).

    .X.X......X
    .X*.X.XXX.X
    .XX.X.XM...
    ......XXXX.

In example above, Ron and Hermione are located at index (2,7) and the portkey is at (1,2).
Each cell is indexed according to Matrix Conventions. Hermione decides it's time to find the portkey
and leave. They start along the path and each time they have to choose a direction, she waves her wand
and it points to the correct direction. Ron is betting that she will have to wave her wand exactly 'k'
times. Can you determine if Ron's guesses are correct?

The map from above has been redrawn with the path indicated as a series where M is the starting point
(no decision in this case), 1 indicates a decision point and 0 is just a step on the path:

    .X.X.10000X
    .X*0X0XXX0X
    .XX0X0XM01.
    ...100XXXX.

There are three instances marked with 1 where Hermione must use her wand.

Function Description:
    Complete the countLuck function in the editor below. It should return a string,
    either  'Impressed'  if Ron is correct or  'Oops!'  if he is not.
    countLuck has the following parameters:
        matrix: a list of strings, each one represents a row of the matrix
        k: an integer that represents Ron's guess

Input Format:
    The first line contains an integer t, the number of test cases.
    Each test case is described as follows:
    The first line contains 2 space-separated integers n and m, the number of forest matrix rows and columns.
    Each of the next n lines contains a string of length m describing a row of the forest matrix.
    The last line contains an integer k, Ron's guess as to how many times Hermione will wave her wand.

Constraints:
    1 <= t <= 10
    1 <= n,m <= 100
    0 <= k <= 10000
    There will be exactly one M and one * in the forest.
    Exactly one path exists between M and *.
"""
import os
from time import time
from collections import defaultdict

TEST1 = ([[['*.M', '.X.'], 1], 'Impressed'],
         [[['.X.X......X', '.X*.X.XXX.X', '.XX.X.XM...', '......XXXX.'], 3], 'Impressed'],
         [[['.X.X......X', '.X*.X.XXX.X', '.XX.X.XM...', '......XXXX.'], 4], 'Oops!'],)


def countLuck(matrix, k):
    matrix = list(map(list, matrix))
    n_elem = len(matrix)
    m_elem = len(matrix[0])
    
    # find entry and exit points
    for idx, line in enumerate(matrix):
        for inner_idx in range(m_elem):
            if line[inner_idx] == 'M':
                start_point = (idx, inner_idx)
            if line[inner_idx] == '*':
                end_point = (idx, inner_idx)
 
    st = [start_point]
    tracker = defaultdict(int)
    tracker[start_point] = 0
 
    # iterate the DFS list
    while st:
        curr = st.pop()
 
        # exit found
        if curr == end_point:
            if tracker[curr] == k:
                return 'Impressed'
            else:
                return 'Oops!'
 
        # save all exits that were not visited before
        inner_st = set()
        if (curr[0] > 0 and (matrix[curr[0]-1][curr[1]] == "." or matrix[curr[0]-1][curr[1]] == "*")) \
                and (curr[0]-1, curr[1]) not in tracker:
            # if the point in the prior line is valid and not in tracker
            inner_st.add((curr[0]-1, curr[1]))
        if (curr[1] > 0 and (matrix[curr[0]][curr[1]-1] == "." or matrix[curr[0]][curr[1]-1] == "*")) \
                and (curr[0], curr[1]-1) not in tracker:
            # if the prior point in the line is valid and not in tracker
            inner_st.add((curr[0], curr[1]-1))
        if (curr[0] < n_elem - 1 and (matrix[curr[0]+1][curr[1]] == "." or matrix[curr[0]+1][curr[1]] == "*")) and \
                (curr[0]+1, curr[1]) not in tracker:
            # if the point in the next line is valid and....
            inner_st.add((curr[0]+1, curr[1]))
        if (curr[1] < m_elem - 1 and (matrix[curr[0]][curr[1]+1] == "." or matrix[curr[0]][curr[1]+1] == "*")) and \
                (curr[0], curr[1]+1) not in tracker:
            # if the next point in the line is valid and....
            inner_st.add((curr[0], curr[1]+1))
 
        # a crossroad
        if len(inner_st) > 1:
            tracker[curr] += 1
 
        # save the nodes to DFS list
        for n in inner_st:
            tracker[n] = tracker[curr]
            st.append(n)
 
    return 'Oops!'


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())

    for t_itr in range(t):
        nm = input().split()
        n = int(nm[0])
        m = int(nm[1])
        matrix = []

        for _ in range(n):
            matrix_item = input()
            matrix.append(matrix_item)
        k = int(input())
        result = countLuck(matrix, k)
        fptr.write(result + '\n')
    fptr.close()


def test():
    data = TEST1

    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = countLuck(input[0], input[1])
        t1 = time() - t0
        print(f'Expected: {output}\t\t Result: {rst}')
        assert (rst == output)
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()