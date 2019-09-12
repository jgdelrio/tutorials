"""
Algorithms > Strings > Common Child

A string is said to be a child of a another string if it can be formed by deleting 0 or more
characters from the other string. Given two strings of equal length, what's the longest string
that can be constructed such that it is a child of both?

For example, ABCD and ABDC have two children with maximum length 3, ABC and ABD.
They can be formed by eliminating either the D or C from both strings. Note that we will
not consider ABCD as a common child because we can't rearrange characters and ABCD != ABDC.

Complete the commonChild function in the editor below. It should return the longest string
which is a common child of the input strings. commonChild has the following parameter(s):
    s1, s2: two equal length strings

Constraints:
    1 <= |s1|, |s2| <= 5000
    All characters are upper case in the range ascii[A-Z].

Output Format:
    Print the length of the longest string s, such that s is a child of both s1 and s2.

Notes:
    Time complexity is O(N*M)
    Space complexity is O(N*M)
    It is a common subsequence problem:
        http://www.geeksforgeeks.org/dynamic-programming-set-4-longest-common-subsequence/
"""
import os
from time import time

TEST1 = (['HARRY', 'SALLY'], 2)     # The longest common string deleting characters is AY
TEST2 = (['AA', 'BB'], 0)
TEST3 = (['SHINCHAN', 'NOHARAAA'], 3)
TEST4 = (['ABCDEF', 'FBDAMN'], 2)


def commonChild(s1, s2):
    # allocate an array of arrays
    len_s1 = len(s1)
    len_s2 = len(s2)
    lengths = [[0] * (len_s1 + 1) for _ in range(len_s2 + 1)]
    for i, x in enumerate(s1):
        for j, y in enumerate(s2):
            # compare each element of s1 with each element of s2
            if x == y:
                # if they are equal, allocate +1
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                # if they are different allocate the max between...
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    # return the last value of the matrix
    return lengths[-1][-1]


def common_child(s1, s2):
    # allocate an array of arrays
    len_s1 = len(s1)
    len_s2 = len(s2)
    # build history of previous row
    lengths = [[0] * (len_s1 + 1) for _ in range(len_s2)]
    for i, x in enumerate(s1):
        # get index pointers for current and previous row
        li1 = (i+1)%2
        li = i%2
        # j in s1 = j+1 in lengths
        for j in range(len_s2):
            # i and j are used to step forward in each string.
            # Now check if s1[i] and s2[j] are equal 
            if s1[i] == s2[j]:
                # Now we have found one longer sequence 
                # than what we had previously found.
                # so add 1 to the length of previous longest
                # sequence which we could have found at
                # earliest previous position of each string,
                # therefore subtract -1 from both i and j
                lengths[li1][j+1] = (lengths[li][j] + 1) 
                #lengths_letters[li1][j+1] = lengths_letters[li][j]+s1[li]

            # if not matching pair, then
            # get the biggest previous value
            elif lengths[li1][j] > lengths[li][j+1]:
                lengths[li1][j+1] = lengths[li1][j] 
                #lengths_letters[li1][j+1] = lengths_letters[li1][j]
            else:
                lengths[li1][j+1] = lengths[li][j+1] 
                #lengths_letters[li1][j+1] = lengths_letters[li][j+1]
    #print(lengths_letters[(i+1)%2][j+1])
    return lengths[(i+1)%2][j+1]


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s1 = input()
    s2 = input()
    result = common_child(s1, s2)
    fptr.write(str(result) + '\n')
    fptr.close()


def test():
    data = TEST3
    input = data[0]
    output = data[1]

    t0 = time()
    rst = commonChild(input[0], input[1])
    t1 = time() - t0
    assert (rst == output)
    print(f'Total time: {t1}')


if __name__ == '__main__':
    test()


