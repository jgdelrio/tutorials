"""
Mathematics > Combinatorics > Merge List

Shashank is very excited after learning about the linked list. He learned about how to merge two linked lists.
When we merge two linked lists, the order of the elements of each list doesn't change.
For example if we merge [1,2,3] and [4,5,6], [1,4,2,3,5,6] is a valid merge, while [1,4,3,2,5,6] is not a valid
merge because 3 appears before 2.

You are given two lists having sizes N and M.
How many ways can we merge both lists? It is given that all N+M elements are distinct.
As your answer can be quite large. The output must be the number of combinations  mod (10e9 + 7)

Input Format:
The first line contains an integer T, the number of test cases.
Each of the next T lines contains two integers N and M.

Constraints:
1 <= T <= 10
1 <= N <= 100
1 <= M <= 100

Output Format:
Print the value of the answer   mod (10e9 + 7)


Notes:
  We can also think as first of all out of n+m places we can choose n places for 1st list,
  so total no. of ways for this is (n+m)Cn. Now since these n digits have a fixed order we cannot
  rearrange them. And we have m places left for m numbers of the 2nd list. Since these m numbers cannot be rearranged
  there is only one way to put them in remaining m places.
  Finnaly that's  (n+m)C(n)*1! *(m)C(m)*1   which is    (n+m)Cm i.e, (n+m)!/(n!*m!)
"""

import os
from math import factorial

# Define input and output
TEST1 = [(2, 2), 6]


def solve(n, m):
    result = (factorial(n + m) // factorial(n) // factorial(m) % (10 ** 9 + 7))
    return result


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())

    for t_itr in range(t):
        nm = input().split()
        n = int(nm[0])
        m = int(nm[1])
        result = solve(n, m)
        fptr.write(str(result) + '\n')

    fptr.close()


def test():
    data = TEST1
    input = data[0]
    output = data[1]

    n = input[0]
    m = input[1]
    result = solve(n, m)

    print("Result: {}".format(result))
    assert(result == output)


if __name__ == '__main__':
    test()
