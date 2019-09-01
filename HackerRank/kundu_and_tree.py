"""
Data Structures > Disjoint Set

Kundu is true tree lover. Tree is a connected graph having N vertices and N-1 edges.
Today when he got a tree, he colored each edge with one of either red(r) or black(b) color.
He is interested in knowing how many triplets(a,b,c) of vertices are there , such that,
there is at least one edge having red color on all the three paths i.e. from vertex a to b,
vertex b to c and vertex c to a .
Note that (a,b,c), (b,a,c) and all such permutations will be considered as the same triplet.

If the answer is greater than 10e9 + 7, print the answer modulo (%) 10e9 + 7.

Input Format:
The first line contains an integer N, i.e., the number of vertices in tree.
The next N-1 lines represent edges: 2 space separated integers denoting an edge followed by a color of the edge.
A color of an edge is denoted by a small letter of English alphabet, and it can be either red(r) or black(b).

Output Format:
Print a single number i.e. the number of triplets mod (10e9 + 7)

Constraints:
1 ≤ N ≤ 10^5
A node is numbered between 1 to N.


Note:
    The idea is suppose given n sets and needed to select 2 elements from them such that both don't
    belong to same set (here we need to find 3 such elements but showing for 2 give the idea).
    If the array A={5,2,3,4,10,7} stores the size of each set.    # of sets=6 and A[0]=5  (0-based indexing)
    Lets state the problem mathematically:
      We need sum of : i * j for all distinct pairs (i,j) where i and j belong to array A.
    You can do it in O(n * n) by 2 loops :
       for i in range(len(A)-1):
          for j in range(i+1, len(a)):
             sum += A[i] * A[j]

    What if instead of 2 loops, we create another array B of same size and store in B[i] the sum
    of all elements from A[i] to A[A.length-1].

    Now we can write sum as:
       for i in range(len(A)-1):
          B[i+1] = sum(A[])
          sum += (A[i] * B[i+1]);

    We were multiplying and then summing... by this way we sum first in one loop, then multiply in another loop!
"""
import os

with open(os.path.abspath(os.path.join('data', 'kundu_and_tree_input08.txt')), 'r') as f:
    n = int(f.readline())
    input_data = []
    for k in range(n-1):
        line = [int(k) if i < 2 else k[0] for i, k in enumerate(f.readline().split(' '))]
        input_data.append(line)

# Define input and output
TEST1 = ([[1, 2, 'b'], [2, 3, 'r'], [3, 4, 'r'], [4, 5, 'b']], 4)
TEST2 = (input_data, 994774931)


class DisjointSet:
    def __init__(self, index):
        self.size = 1
        self.index = index
        self.parent = self

    def find_root(self):
        if self.parent != self:
            parent = self.parent.find_root()
            return parent
        else:
            return self.parent

    def union(self, other):
        if other == self:
            return
        root = self.find_root()
        other_root = other.find_root()

        if other_root == root:
            return

        # union by size
        if root.size >= other_root.size:
            other_root.parent = root
            root.size += other_root.size
        else:
            root.parent = other_root
            other_root.size += root.size


def create_components(disjoint_set, index):
    if disjoint_set[index] is None:
        disjoint_set[index] = DisjointSet(index)
    return disjoint_set[index]


def possible_triplets(n):
    if n < 3:
        return 0
    res = 1
    for i in range(n-2, n+1):
        res *= i
    return res / 6


def possible_pairs(n):
    #  n! / (k! (n-k)!) = (n - k + 1) * ... * n / k! = (n - 1) * ... * n / 2
    if n < 2:
        return 0

    res = 1
    for i in range(n-1, n+1):
        res *= i
    return res / 2


def count_triplets(tree, color='r'):
    """
    Return the total number of red triplets
    :param tree:
    :return:
    """
    n = len(tree) + 1
    components = [None] * (n+1)

    for c in tree:
        if c[2] == color:
            continue
        # Only retain the opposite color
        node_a = create_components(components, c[0])
        node_b = create_components(components, c[1])
        node_a.union(node_b)

    # make elements unique
    unique_components = []
    for c in components:
        if c is not None:
            unique_components.append(c.find_root())
    unique_components = set(unique_components)

    valid_triplets = possible_triplets(n)

    for c in unique_components:
        # subtract all triplets within the components where I have two or more black edges
        valid_triplets -= possible_triplets(c.size)

        # subtract all triplets build from 2 vertices of the components and 1 other vertex
        valid_triplets -= possible_pairs(c.size) * (n - c.size)

    rst = int(valid_triplets % (10e9 + 7))
    return rst


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
    data = TEST2
    tree = data[0]
    output = data[1]

    result = count_triplets(tree)
    print("N Triplets: {}".format(result))
    assert(result == output)


if __name__ == '__main__':
    test()
