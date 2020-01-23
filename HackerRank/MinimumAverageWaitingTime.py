"""
Practice > Data Structures > Heap > Minimum Average Waiting Time

Tieu owns a pizza restaurant and he manages it in his own way. While in a normal restaurant,
a customer is served by following the first-come, first-served rule, Tieu simply minimizes
the average waiting time of his customers. So he gets to decide who is served first,
regardless of how sooner or later a person comes.

Different kinds of pizzas take different amounts of time to cook. Also, once he starts cooking a pizza,
he cannot cook another pizza until the first pizza is completely cooked. Let's say we have three customers
who come at time t=0, t=1, & t=2 respectively, and the time needed to cook their pizzas is 3, 9, & 6 respectively.
If Tieu applies first-come, first-served rule, then the waiting time of three customers is 3, 11, & 16 respectively.
The average waiting time in this case is (3 + 11 + 16) / 3 = 10. This is not an optimized solution.
After serving the first customer at time t=3, Tieu can choose to serve the third customer.
In that case, the waiting time will be 3, 7, & 17 respectively. Hence the average waiting time is (3 + 7 + 17) / 3 = 9.

Help Tieu achieve the minimum average waiting time.
For the sake of simplicity, just find the integer part of the minimum average waiting time.

Input Format
------------
- The first line contains an integer N, which is the number of customers.
- In the next N lines, the ith line contains two space separated numbers Ti and Li.
  Ti is the time when ith customer order a pizza, and Li is the time required to cook that pizza.
- The ith customer is not the customer arriving at the ith arrival time.

Note
----
The waiting time is calculated as the difference between the time a customer orders pizza
(the time at which they enter the shop) and the time she is served.

Cook does not know about the future orders.
"""

import os
from time import time
from heapq import heappush, heappop


TESTS = [
    ([[0, 3], [1, 9], [2, 6]], 9),
    ([[0, 3], [1, 9], [2, 5]], 8)
]


def minimumAverage(customers):
    customers.sort(reverse=True)
    n_customers = len(customers)

    pqueue = []
    time_waiting = 0
    current_time = 0

    while customers or pqueue:
        while customers and customers[-1][0] <= current_time:
            # At first only consider in the queue customers available at time 0
            # Only add more customers to the processing queue if the current_time is bigger or equal than arrival
            heappush(pqueue, customers.pop()[::-1])

        if pqueue:
            # Store current time with length of task as that's when it will be ready to consider next order
            current_task = heappop(pqueue)
            current_time += current_task[0]
            time_waiting += current_time - current_task[1]
        else:
            # No processing queue. Force filling proc. queue and incresase current time
            heappush(pqueue, customers.pop()[::-1])
            current_time = pqueue[0][1]

    min_av_waiting_time = (time_waiting // n_customers)
    return min_av_waiting_time


def hackerrank_run():
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    customers = []

    for _ in range(n):
        customers.append(list(map(int, input().rstrip().split())))

    result = minimumAverage(customers)
    fptr.write(str(result) + '\n')
    fptr.close()


def test():
    data = TESTS

    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = minimumAverage(input)
        t1 = time() - t0
        assert (rst == output)
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
