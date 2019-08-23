"""
Algorithms > Strings

Calculate the string similarity:
For a given string (ex: abaab) taking substring starting from the end ('b', 'ab', 'aab', 'baab' and 'abaab')
sum the number of characters of the substring that are equal to the beginning of the initial string not 
adding from the moment in which the substring differs. And give the final result.

In the previous example that will be:
- For ('b', 'ab', 'aab', 'baab' and 'abaab') 
- [0, 2, 1, 0, 5]  -->  Similarity = 8
"""

import os
from time import time
import timeit

TESTS = [('aa', 3),
         ('ababaa', 11),
         ('aaaabaaa', 20),
         ('aapcaaabaaaaapcaaaaaaapcaabbbaapcappppaaaaaaaapcaba', 118),
         ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
         17589)]


# Complete the stringSimilarity function below.
def string_similarity_1(s):
    """Straight comparison"""
    similarity = len(s)
    for r in range(similarity-1):
        current = s[-(r+1):]
        clength = len(current)
        if current[0] != s[0]:
            continue
        elif current == s[:clength]:
            similarity += clength
            continue
        else:
            cn = 2
            while (current[:cn] == s[:cn]) and (cn <= clength):
                cn += 1
            similarity += cn - 1
    return similarity


def string_similarity_2(s):
    """Straight comparison with halving indexing"""
    similarity = len(s)
    for r in range(similarity-1):
        current = s[-(r+1):]
        clength = len(current)

        if current[0] != s[0]:
            continue
        elif current == s[:clength]:
            similarity += clength
            continue
        else:
            idx = round(clength / 2)
            part = idx
            while current[:idx] == s[:idx] and part > 0:
                part = round(part/2)
                idx += part
            while current[:idx] != s[:idx]:
                idx -= 1

            similarity += idx

    return similarity


def string_similarity_3(s):
    """Using chars..."""
    n = len(s)
    similarity = [0] * n
    similarity[0] = 1

    for r in range(1, n):
        adding = [s[r] == y for y in s[:r+1]]
        similarity[r] += adding[0]
        adding.reverse()
        try:
            idx = adding.index(False)
            rr = min(idx, r)
        except ValueError:
            rr = r

        # similarity = [*[similarity[x] + adding[x] if similarity[x]>0 else 0 for x in range(rr)], *similarity[rr:]]
    # print(similarity)
    return sum(similarity)


def string_similarity(s):
    """Char method"""
    n = len(s)
    similarity = [0] * n
    left, right = 0, 0

    for i in range(n):
        # for each letter
        if i > right:
            # make equal all indices
            left = right = i
            # compare each letter with all previous letters (max similarity is the total length)
            while right < n and (s[right] == s[right - left]):
                right += 1
            # similarity as the difference
            similarity[i] = right - left
            right -= 1
        else:
            k = i - left
            if similarity[k] < (right - i + 1):
                similarity[i] = similarity[k]
            else:
                left = i

                while right < n and (s[right] == s[right - left]):
                    right += 1
                similarity[i] = right - left
                right -= 1

    return sum(similarity) + n


def test():
    """Compare the performance of the 3 stringSimilarity functions"""
    setup = 'from hack import stringSimilarity, stringSimilarity1, stringSimilarity2; s="aapcaaabaaaaapcaaaaaaapcaabbbaapcappppaaaaaaaapcaba"'

    rst = timeit.repeat(stmt='string_similarity(s)',
                        setup=setup,
                        number=1000,
                        repeat=50)
    print('Timing final -> max {:.4f} \t min {:.4f} \t av {:.4f}'.format(max(rst), min(rst), sum(rst)/len(rst)))

    rst = timeit.repeat(stmt='string_similarity_1(s)',
                        setup=setup,
                        number=1000,
                        repeat=50)
    print('Timing 1 \t-> max {:.4f} \t min {:.4f} \t av {:.4f}'.format(max(rst), min(rst), sum(rst)/len(rst)))

    rst = timeit.repeat(stmt='string_similarity_2(s)',
                        setup=setup,
                        number=1000,
                        repeat=50)
    print('Timing 2 \t-> max {:.4f} \t min {:.4f} \t av {:.4f}'.format(max(rst), min(rst), sum(rst)/len(rst)))


if __name__ == '__main__':
    t0 = time()
    for e in TESTS:
        s, rst = e
        result = string_similarity(s)

        print('Result: {} \t Expected: {}'.format(result, rst))

    print('Time: {}'.format(time()-t0))
