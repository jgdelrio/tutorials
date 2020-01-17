"""
Practice > Data Structures > Linked Lists > Inserting a Node Into a Sorted Doubly Linked List

Given a reference to the head of a doubly-linked list and an integer, 'data',
create a new DoublyLinkedListNode object having data value 'data' and insert it
into a sorted linked list while maintaining the sort.

The function to complete has two parameters:
- head: A reference to the head of a doubly-linked list of DoublyLinkedListNode objects.
- data: An integer denoting the value of the  field for the DoublyLinkedListNode you must insert into the list.

Note: Recall that an empty list (i.e., where head=null) and a list with one element are sorted lists.

Do not print anything to stdout. Your method must return a reference to the  of the same list that was passed to it as a parameter.


Input Format
------------
The first line contains an integer t, the number of test cases.

Each of the test case is in the following format:

The first line contains an integer n, the number of elements in the linked list.
Each of the next n lines contains an integer, the data for each node of the linked list.
The last line contains an integer  which needs to be inserted into the sorted doubly-linked list.
"""

import os
from time import time

TESTS = [[([1, 3, 4, 10], 5), [1, 3, 4, 5, 10]]]


class Node:
    def __init__(self):
        self.data = None
        self.prev = None
        self.next = None


def doublelinkedlist(array):
    # Receives a sorted array creates the nodes and link them
    base = Node()
    base.data = array[0]

    node_0 = base
    for val in array[1:]:
        node_1 = Node()
        node_1.data = val
        # Link nodes
        node_1.prev, node_0.next = node_0, node_1
        node_0 = node_1

    return base


def data_from_d_linked_list(dlist):
    data = [dlist.data]
    while dlist.next is not None:
        data.append(dlist.next.data)
        dlist = dlist.next
    return data


def sortedInsert(head, data):
    new_node = Node()
    new_node.data = data

    if new_node.data < head.data:       # If data < 0    attach the previous head to the new node
        new_node.next = head
        head.prev = new_node
        return new_node

    else:                               # data >= 0
        node_0 = head                   # Take base current node and create link to next node
        node_1 = head.next
        while (node_1 is not None) and (new_node.data > node_1.data):
            # Keep moving through the linked list while data is bigger than next node
            node_0 = node_1
            node_1 = node_1.next

        if node_1 is None:
            # Once reached the desired location, check if next node is null
            # in that case just link new with previous as new becomes the tail of the d-linked list
            node_0.next = new_node
            new_node.prev = node_0
        else:
            # Or if not null then link
            node_0.next = new_node        # with previous
            new_node.prev = node_0
            node_1.prev = new_node        # and with following
            new_node.next = node_1

        node_0 = head

        while node_0 is not None:
            node_0 = node_0.next
        return head


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())

    for t_itr in range(t):
        llist_count = int(input())
        llist = DoublyLinkedList()
        for _ in range(llist_count):
            llist_item = int(input())
            llist.insert_node(llist_item)
        data = int(input())
        llist1 = sortedInsert(llist.head, data)
        print_doubly_linked_list(llist1, ' ', fptr)
        fptr.write('\n')
    fptr.close()


def test():
    data = TESTS

    for d in data:
        input = d[0]
        output = d[1]

        double_linked_list = doublelinkedlist(input[0])

        t0 = time()
        rst = sortedInsert(double_linked_list, input[1])
        t1 = time() - t0
        assert (data_from_d_linked_list(rst) == output)
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
