"""


Complete the preOrder function in your editor below, which has 1 parameter:

a pointer to the root of a binary tree.
It must print the values in the tree's preorder traversal as a single line of space-separated values.

Input Format:
Our hidden tester code passes the root node of a binary tree to your preOrder function.

Constraints:
1 <= Nodes in the tree <= 500

Sample:
     1
      \
       2
        \
         5
        /  \
       3    6
        \
         4

Node is defined as:
    self.left (the left child of the node)
    self.right (the right child of the node)
    self.info (the value of the node)
"""

# Define input and output
TEST1 = ([6, "1 2 5 3 6 4"], "1 2 5 3 4 6")


def next_move(result, node):
    if node.left is not None:
        result.append(node.left.info)
        result = next_move(result, node.left)
    if node.right is not None:
        result.append(node.right.info)
        result = next_move(result, node.right)
    return result


def preOrder(root):
    """Receive the root of the tree"""
    result = []
    if root.info is None:
        print('')
    else:
        result.append(root.info)
        result = next_move(result, root)

    print(' '.join(map(str, result)))

