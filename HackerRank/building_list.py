"""
Mathematics > Combinatorics > Building a List

Chan has decided to make a list of all possible combinations of letters of a given string S.
If there are 2 strings with the same set of characters,
print the lexicographically smallest arrangement of the 2 strings.

    abc acb cab bac bca

all the above strings' lexicographically smallest string is abc.
Each character in the string S is unique. Your task is to print the entire list of Chan's in lexicographic order.
for string abc, the list in lexicographic order is given below

    a ab abc ac b bc c

Input Format:
The first line contains the number of test cases T. T testcases follow.
Each testcase has 2 lines. The first line is an integer N ( the length of the string).
The second line contains the string S.

Output Format:
For each testcase print the entire list of combinations of string S, with each combination of letters in a newline.

Constraints:
0< T< 50
1< N< 16
string S contains only small alphabets(a-z)

Relevant Documentation:   http://www.keithschwarz.com/binary-subsets/
"""

import os
from itertools import combinations

# Define input and output
TEST1 = ([(2, 'ab'), ['a', 'ab', 'b']], ((3, 'xyz'), ['x', 'xy', 'xyz', 'xz', 'y', 'yz', 'z']))


def lexicographic(s):
    """Implementation using the itertools combinations function"""
    build = []
    for i in range(len(s)):
        build.extend(map(''.join, combinations(s, i+1)))
    build.sort()
    return build


def lexic_02(s, p1, p2, lev):
    """This implementation miss some combinations"""
    build = []
    current = ""
    if p2 == len(s) and p1 != p2:
        lev += 1
        p2 = p1 = lev
    if p2 == p1 == len(s):
        return []
    else:
        for x in range(p1, p2 + 1):
            current += s[x]
        build.append(current)

        build.extend(lexic_02(s, p1, p2 + 1, lev))

    return build


def lexic_03(s):
    """Implementation building the combinations internally"""
    n = len(s)
    build = []

    for x in range(2 ** n):         # The combinations of n characters is 2^n
        # First we build the binary representation of the combinations (000, 001, 010, 011, 111)
        b = str(("%0" + str(n) + "d") % int(bin(x)[2:]))
        r = ''
        for y in range(n):
            if b[y] == '1':     # Add characters of activated bits
                r += s[y]
        if len(r) > 0:
            build.append(r)
    return sorted(build)


def solve(s):
    rst = lexicographic(sorted(s))
    # rst = lexic_02(''.join(sorted(s)), 0, 0, 0)
    # rst = lexic_03(''.join(sorted(s)))
    return rst


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())

    for t_itr in range(t):
        n = int(input())
        s = input()
        result = solve(s)
        fptr.write('\n'.join(result))
        fptr.write('\n')

    fptr.close()


def test():
    data = TEST1
    for k in data:
        input = k[0]
        output = k[1]

        s = input[1]
        result = solve(s)

        print("Result: {}".format(result))
        assert(result == output)


if __name__ == '__main__':
    test()
