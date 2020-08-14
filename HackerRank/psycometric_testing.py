"""
Psychometric testing is designed to find job-relevant information about an applicant that the traditional
interview process wouldn't otherwise uncover. It typically includes a combination of online aptitude and
personality tests that measure cognitive ability and personality traits.

A company has psychometric scores for n candidates, and it will only extend job offers to candidates with
scores in the inclusive range given by  [loverLimit, upperLimit]. Given the list of scores and a sequence
of score ranges, determine how many candidates the company will extend offers to for each range of scores.

Complete the jobOffers function which has three parameters:

1. An array of n integers, scores. denoting the list of candidate scores.
2. An array of q integers, lowerLimits, where each lowerLimits(i) denotes the lowerLimit for score range i.
3. An array of q integers, upperLimits, where each upperLimits(i) denotes the upperLimit for score range i.

The function must return an array of q integers where the value at each index i denotes the number
of candidates in the inclusive range that the company will extend offers to.

Example:
    With scores = [1, 3, 5, 6, 8], lowerLimits = [2] and upperLimits = [6]     -->  Result: 3
"""

import os
from time import time
from collections import Counter
from itertools import combinations


TESTS = [
    (([1, 3, 5, 6, 8], [2], [6]), 3),
    (([4, 8, 7], [2, 4], [8, 4]), [3, 1]),
]


def jobOffers(scores: list, lowerLimits: list, upperLimits: list) -> list:
    job_offers = [0] * len(lowerLimits)

    scores.sort()
    for s in scores:
        for i, (l, u) in enumerate(zip(lowerLimits, upperLimits)):
            if (s >= l) and (s <= u):
                job_offers[i] += 1
    return job_offers


def hackerrank_run():
    scores = []
    lowerLimits = []
    upperLimits = []

    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())
    for n_score in range(n):
        scores.append(int(input()))
    q = int(input())
    for q_limit in range(q):
        lowerLimits.append(int(input()))
    q = int(input())
    for q_limit in range(q):
        upperLimits.append(int(input()))

    result = jobOffers(scores, lowerLimits, upperLimits)
    fptr.write(str(result) + '\n')
    fptr.close()


def test():
    data = TESTS
    for d in data:
        input = d[0]
        output = d[1]

        t0 = time()
        rst = jobOffers(*input)
        t1 = time() - t0
        if not rst == output:
            print(f"{input} Expected: {output} Result: {rst}")
        print(f'Total time: {t1}')


if __name__ == '__main__':
    test()
