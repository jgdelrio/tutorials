"""
# Practice > Mathematics > Fundamentals > Special Multiple

You are given an integer N. Can you find the least positive integer X made up of only 9's and 0's, 
such that, X is a multiple of N?

Update
X is made up of one or more occurences of 9 and zero or more occurences of 0.

Input Format 
The first line contains an integer T which denotes the number of test cases. T lines follow. 
Each line contains the integer N for which the solution has to be found.

Output Format 
Print the answer X to STDOUT corresponding to each test case. The output should not contain any leading zeroes.

Constraints 
1 <= T <= 104 
1 <= N <= 500
"""

import os
from time import time
# Create an empty queue
from queue import Queue

TEST1 = ([5, 90],
         [7, 9009],
         [1, 9],)


def binary_generator(n):
    """This impolementation will time out in the last test"""
    q = Queue()
    # Enqueu the 1st binary number
    q.put("1")

    # This loop is like BFS of a tree with 1 as root
    # 0 as left child and 1 as right child and so on
    while True:
        s1 = q.get()
        s2 = s1  # Store s1 before changing it

        # Append "0" to s1 and enqueue it
        q.put(s1 + "0")

        # Append "1" to s2 and enqueue it. Note that s2
        # contains the previous front
        q.put(s2 + "1")
        yield int(s1.replace('1', str(n)))


def binary_generator2(n):
    """Either of this implementations are ok"""
    top = int(9e99)
    for k in range(1, top):
        yield n * int(bin(k)[2:])       # n * int(f'{k:0>b}')


# Complete the solve function below.
def solve(n):
    my_909_generator = binary_generator2(9)
    limit = 10e8

    while limit:
        limit -= 1
        posible_solution = next(my_909_generator)
        if posible_solution % n == 0:
            return posible_solution


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())

    for t_itr in range(t):
        n = int(input())
        result = solve(n)
        fptr.write(str(result) + '\n')

    fptr.close()


def test():
    data = TEST1
    repeat = int(1e5)
    t = []

    # for r in range(repeat):
    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = solve(input)
        t.append(time() - t0)
        assert (rst == output)
    print(f'Av time: {sum(t) / repeat}')
    print(f'Total time: {sum(t)}')


def test_binany_generation():
    repeat = int(1e3)
    t = []
    for r in range(repeat):
        t0 = time()
        a = binary_generator2(9)
        for k in range(repeat):
            next(a)
        t.append(time() - t0)
    print(f'Av time: {sum(t) / repeat}')
    print(f'Total time: {sum(t)}')


if __name__ == '__main__':
    test()
