"""
Data Structures > Trees > Is This a Binary Search Tree?

For the purposes of this challenge, we define a binary tree to be a binary search tree
with the following ordering requirements:
    - The  value of every node in a node's left subtree is less than the data value of that node.
    - The  value of every node in a node's right subtree is greater than the data value of that node.
    - Given the root node of a binary tree, can you determine if it's also a binary search tree?

Complete the function with 'l' parameter: a pointer to the root of a binary tree. It must return
a boolean denoting whether or not the binary tree is a binary search tree. You may have to write
one or more helper functions to complete this challenge.

Input Format:
    You are not responsible for reading any input from stdin. Hidden code stubs will assemble a
    binary tree and pass its root node to your function as an argument.

Constraints:
    0 <= data <= 1e4

Output Format:
    You are not responsible for printing any output to stdout. Your function must return true if
    the tree is a binary search tree; otherwise, it must return false. Hidden code stubs will print
    this result as a Yes or No answer on a new line.

"""
from time import time


def min_val(tree):
    if tree.left is None:
        min_left = tree.data
    else:
        min_left = min_val(tree.left)

    if tree.right is None:
        min_right = tree.data
    else:
        min_right = min_val(tree.right)

    val = min(tree.data, min_left, min_right)
    return val


def max_val(tree):
    if tree.left is None:
        min_left = tree.data
    else:
        min_left = max_val(tree.left)

    if tree.right is None:
        min_right = tree.data
    else:
        min_right = max_val(tree.right)

    val = max(tree.data, min_left, min_right)
    return val


def naive_check_binary_search_tree(node):
    """
    This runs slowly since it crosses some parts of the tree many times
    """
    if node is None:
        return True
    if node.left is not None and \
            max_val(node.left) > node.data:
        return False
    if node.right is not None and \
            min_val(node.right) < node.data:
        return False
    if not naive_check_binary_search_tree(node.left) or \
            not naive_check_binary_search_tree(node.right):
        return False
    return True


INT_MIN = -float('Inf')
INT_MAX = float('Inf')


class Node:
    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


def is_BST_tool(node, v_min, v_max):
    # the tree is empty is BST
    if node is None:
        return True

    # The node violates the min or max constrain
    if node.data < v_min or node.data > v_max:
        return False

    # Otherwise check the subtrees recursively
    return (is_BST_tool(node.left, v_min, node.data - 1) and    # all data in left tree must be smaller
            is_BST_tool(node.right, node.data + 1, v_max))      # all data in right tree must be greater


def check_binary_search_tree(node):
    """
    The trick is to use a helper function is_BST_tool(node, v_min, v_max) that traverses down
    the tree keeping track of the narrowing min and max allowed values as it goes, looking at
    each node only once. The initial values for min and max should be INT_MIN and INT_MAX
    """
    return is_BST_tool(node, INT_MIN, INT_MAX)


root = Node(4)
root.left = Node(2)
root.right = Node(5)
root.left.left = Node(1)
root.left.right = Node(3)

TEST = [root, True]


def test():
    data = TEST

    input = data[0]
    output = data[1]

    t0 = time()
    rst = check_binary_search_tree(input)
    t1 = time() - t0
    assert (rst == output)
    print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
