"""
Kundu is true tree lover. Tree is a connected graph having N vertices and N-1 edges.
Today when he got a tree, he colored each edge with one of either red(r) or black(b) color.
He is interested in knowing how many triplets(a,b,c) of vertices are there , such that,
there is at least one edge having red color on all the three paths i.e. from vertex a to b,
vertex b to c and vertex c to a .
Note that (a,b,c), (b,a,c) and all such permutations will be considered as the same triplet.

If the answer is greater than 10^9 + 7, print the answer modulo (%) 10^9 + 7.

Input Format:
The first line contains an integer N, i.e., the number of vertices in tree.
The next N-1 lines represent edges: 2 space separated integers denoting an edge followed by a color of the edge.
A color of an edge is denoted by a small letter of English alphabet, and it can be either red(r) or black(b).

Output Format:
Print a single number i.e. the number of triplets.

Constraints:
1 ≤ N ≤ 10^5
A node is numbered between 1 to N.

"""
import os

# Define input and output
TEST1 = ([[1, 2, 'b'], [2, 3, 'r'], [3, 4, 'r'], [4, 5, 'b']], 4)


def count_triplets(tree, color='r'):
    """
    Return the total number of red triplets
    :param tree:
    :return:
    """
    n = len(tree)

    def perm_grp_x(n, k=3):
        return n * (n-1)*(n-2) / 6

    # total number of permutations of groups of 3:   n! / (n-3) 3!
    total_triplets = perm_grp_x(n)

    count_oposite = 0
    for i in range(n):
        if tree[i][2] != color:
            count_oposite += 1

    total_triplets_oposite_color = perm_grp_x(count_oposite)

    # count the total number of bad triplets
    # total_bad_triplets = 0
    # for i in range(n):
    #     if tree[i][2] == 'b':
    #         total_bad_triplets += (i-1) * i * (3 * n - 2 * (i+1) - 2)
    # total_bad_triplets /= 6
    return int(total_triplets - total_triplets_oposite_color)


def to_int(x):
    try:
        return int(x)
    except:
        return x


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())

    tree = []
    for i in range(n-1):
        e = list(map(to_int, input().rstrip().split()))
        tree.append(e)

    result = count_triplets(tree)
    fptr.write(str(result) + '\n')
    fptr.close()


def test():
    data = TEST1
    tree = data[0]
    output = data[1]

    result = count_triplets(tree)
    print("N Triplets: {}".format(result))
    assert(result == output)


if __name__ == '__main__':
    test()
