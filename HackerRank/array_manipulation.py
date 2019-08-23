"""


Starting with a 1-indexed array of zeros and a list of operations, for each operation add a value
to each of the array element between two given indices, inclusive. Once all operations have been
performed, return the maximum value in your array.

For example, the length of your array of zeros n=10. Your list of queries is as follows:
    a b k
    1 5 3
    4 8 7
    6 9 1

Add the values of 'k' between the indices 'a' and 'b' inclusive:
index->	 1 2 3  4  5 6 7 8 9 10
	    [0,0,0, 0, 0,0,0,0,0, 0]
	    [3,3,3, 3, 3,0,0,0,0, 0]
    	[3,3,3,10,10,7,7,7,0, 0]
    	[3,3,3,10,10,8,8,8,1, 0]

The largest value is 10 after all operations are performed.
"""
#!/bin/python3
from sys import stdin

# Define input and output
TEST1 = ([5, ["1 2 100", "2 5 100", "3 4 100"]], 200)


def read_stdin():
    (n, operations) = [int(i) for i in stdin.readline().split()]
    return n, operations


def stdin_queries(operations):
    queries = []
    for _ in range(operations):
        queries.append(stdin.readline())
    return queries


def hackerrank_run():
    """Run this function in HackerRank as they provide the input for their tests from stdin"""
    n, operations = read_stdin()
    queries = stdin_queries(operations)
    rst = array_manip(n, queries)
    print(rst)


def array_manip(n, queries):
    # initialize array
    array = [0] * n

    for q in queries:
        (a, b, k) = [int(i) for i in q.split()]
        # Accumulate along the array...
        # adding at the beginning of the change and substracting at the end
        array[a-1] += k
        if b < n:
            array[b] -= k

    prev = 0
    for i in range(len(array)):
        prev += array[i]
        array[i] = prev

    return max(array)


if __name__ == '__main__':
    assert(array_manip(*TEST1[0]) == TEST1[1])      # assert result equals expected output
