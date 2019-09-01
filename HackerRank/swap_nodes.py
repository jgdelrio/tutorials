"""
Data Structures > Trees > Swap Nodes[Algo]

A binary tree is a tree which is characterized by one of the following properties:
- It can be empty (null).
- It contains a root node only.
- It contains a root node with a left subtree, a right subtree, or both. These subtrees are also binary trees.

In-order traversal is performed as
1. Traverse the left subtree.
2. Visit root.
3. Traverse the right subtree.
For this in-order traversal, start from the left child of the root node and keep exploring the left
subtree until you reach a leaf. When you reach a leaf, back up to its parent, check for a right child
and visit it if there is one. If there is not a child, you've explored its left and right subtrees
fully. If there is a right child, traverse its left subtree then its right in the same manner.
Keep doing this until you have traversed the entire tree. You will only store the values of a node
as you visit when one of the following is true:
- it is the first node visited, the first time visited
- it is a leaf, should only be visited once
- all of its subtrees have been explored, should only be visited once while this is true
- it is the root of the tree, the first time visited

Swapping: Swapping subtrees of a node means that if initially node has left subtree L and right subtree R,
   then after swapping, the left subtree will be R and the right subtree, L.

Input Format:
The first line contains n, number of nodes in the tree.
Each of the next n lines contains two integers, a b, where a is the index of left child,
and b is the index of right child of ith node.
Note: -1 is used to represent a null node.
The next line contains an integer, t, the size of .
Each of the next t lines contains an integer , each being a value .

Output Format:
For each k, perform the swap operation and store the indices of your in-order traversal to your result array.
After all swap operations have been performed, return your result array for printing.

Constraints:
  root indes is always 1
  1 <= n <= 1024
  1 <= t <= 100
  1 <= k <= n
  Either   a = -1   or   2 <= a <= n
  Either   b = -1   or   2 <= b <= n
  The index of a non-null child will always be greater than that of its parent.
"""

import os
from collections import deque


# Define input and output
#   The input include the nodes (left-right) and the swap operations
TEST1 = (([[2, 3], [-1, -1], [-1, -1]], [1, 1]),
         [[3, 1, 2], [2, 1, 3]])
TEST2 = (([[2, 3], [-1, 4], [-1, 5], [-1, -1], [-1, -1]], [2]),
         [[4, 2, 1, 5, 3]])
TEST3 = (([[2, 3], [4, -1], [5, -1], [6, -1], [7, 8], [-1, 9], [-1, -1], [10, 11], [-1, -1], [-1, -1], [-1, -1]], [2, 4]),
         [[2, 9, 6, 4, 1, 3, 7, 5, 11, 8, 10], [2, 6, 9, 4, 1, 3, 7, 5, 10, 8, 11]])


class Node:
    def __init__(self, d):
        self.data = d


def build_tree(indexes):
    f = lambda x: None if x == -1 else Node(x)
    children = [list(map(f, x)) for x in indexes]
    nodes = {n.data: n for n in filter(None, sum(children, []))}
    nodes[1] = Node(1)
    for idx, child_pair in enumerate(children):
        nodes[idx + 1].left = child_pair[0]
        nodes[idx + 1].right = child_pair[1]
    return nodes[1]


def inorder(root):
    stack = []
    curr = root
    while stack or curr:
        if curr:
            stack.append(curr)
            curr = curr.left
        elif stack:
            curr = stack.pop()
            yield curr.data
            curr = curr.right


def swap_nodes(indexes, queries):
    root = build_tree(indexes)
    for k in queries:
        h = 1
        q = deque([root])
        while q:
            for _ in range(len(q)):
                node = q.popleft()
                if h % k == 0:
                    node.left, node.right = node.right, node.left
                q += filter(None, (node.left, node.right))
            h += 1
        yield inorder(root)


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())

    indexes = []
    for _ in range(n):
        indexes.append(list(map(int, input().rstrip().split())))

    queries_count = int(input())
    queries = []

    for _ in range(queries_count):
        queries_item = int(input())
        queries.append(queries_item)

    result = swap_nodes(indexes, queries)

    fptr.write('\n'.join([' '.join(map(str, x)) for x in result]))
    fptr.write('\n')
    fptr.close()


def test():
    data = TEST3
    indexes = data[0][0]
    queries = data[0][1]
    output = data[1]

    output_gen = swap_nodes(indexes, queries)
    rst = []
    for n in output_gen:
        n_vals = []
        for v in n:
            n_vals.append(v)
        rst.append(n_vals)
    print("FInal output".format(rst))
    assert(rst == output)


if __name__ == '__main__':
    test()
