import numpy as np
import scipy.stats as ss


def get_indexes(items, searchlist):
    # Find the index of items in the searchlist, if available
    if not isinstance(items, (list, np.ndarray)):
        raise TypeError('items must be a list')
    if not isinstance(searchlist, (list, np.ndarray)):
        raise TypeError('searchlist must be a list')

    # [[i for i, x in enumerate(searchlist) if x == e] for e in items]

    def search_by_index(x):
        try:
            return searchlist.index(x)
        except ValueError:
            return None

    return list(map(search_by_index, items))


def get_index(items, searchlist):
    # Find the index of items in the searchlist, if available
    if not isinstance(items, (list, np.ndarray)):
        raise TypeError('items must be a list')
    if not isinstance(searchlist, (list, np.ndarray)):
        raise TypeError('searchlist must be a list')

    searchlist_as_dict = dict(zip(searchlist, range(0, len(searchlist))))

    def search_elem(x):
        try:
            return searchlist_as_dict[x]
        except KeyError:
            return None

    return list(map(search_elem, items))


def get_rank(items):
    """
    Return the rank starting at 1 (as R's rank function)
    :param items: list of items
    :return:      ranks
    """
    if not isinstance(items, (list, np.ndarray)):
        raise TypeError('items must be a list')

    return ss.rankdata(items)


def dcg_at_k(scores, k=10, method=0):
    """
    Discounted Cumulative Gain
    :param scores: positive real values of relevance scores (list or numpy) in rank order
    :param k:      number of results to consider
    :param method:
    :return:
    """
    if k == -1:
        # Use all: calculate idcg (Ideal Discounted Cumulative Gain)
        s = np.asfarray(scores)
    else:
        s = np.asfarray(scores)[:k]

    if s.size:
        if method == 0:
            return s[0] + np.sum(s[1:] / np.log2(np.arange(2, s.size + 1)))
        elif method == 1:
            return np.sum(s / np.log2(np.arange(2, s.size + 2)))
        elif method == 2:
            # As defined by Kaggle: https://www.kaggle.com/wiki/NormalizedDiscountedCumulativeGain
            return np.sum(np.subtract(np.power(2, s), 1) / np.log2(np.arange(2, s.size + 2)))
        else:
            raise ValueError('method must be 0, 1 or 2.')
    return 0.


def prd_ndcg_at_k(predicted_scores, real_scores, k=10, method=0, idcg=False):
    """
    Normalized Discounted Cumulative Gain
    :param predicted_scores: positive real values of relevance predicted scores (list or numpy) in rank order
    :param real_scores:      positive real values of relevance real scores (list or numpy) in rank order
    :param k:                number of results to consider
    :param method:
    :param idcg:             boolean. Apply Idcg or not.
    :return:
    """
    if idcg:
        dcg_max = dcg_at_k(sorted(real_scores, reverse=True), -1, method)
    else:
        dcg_max = dcg_at_k(sorted(real_scores, reverse=True), k, method)
    if not dcg_max:
        return 0.

    return dcg_at_k(predicted_scores, k, method) / dcg_max


def mean_binary_reciprocal_rank(rs):
    """
    Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
            rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
            mean_binary_reciprocal_rank(rs)
                0.61111111111111105
            rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
            mean_binary_reciprocal_rank(rs)
                0.5
            rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
            mean_binary_reciprocal_rank(rs)
                0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_binary_precision(r):
    """
    Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
            r = [0, 0, 1]
            r_binary_precision(r)
                0.33333333333333331
            r = [0, 1, 0]
            r_binary_precision(r)
                0.5
            r = [1, 0, 0]
            r_binary_precision(r)
                1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def binary_precision_at_k(r, k):
    """
    Score is precision @ k. Relevance is binary (nonzero is relevant).
            r = [0, 0, 1]
            binary_precision_at_k(r, 1)
                0.0
            binary_precision_at_k(r, 2)
                0.0
            binary_precision_at_k(r, 3)
                0.33333333333333331
            binary_precision_at_k(r, 4)
                Traceback (most recent call last):
                    File "<stdin>", line 1, in ?
                ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: level of relevant elements
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_binary_precision(r):
    """
    Score is average precision (area under PR curve). Relevance is binary (nonzero is relevant).
            r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
            delta_r = 1. / sum(r)
            sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
                0.7833333333333333
            average_precision(r)
                0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [binary_precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_binary_average_precision(rs):
    """
    Score is mean average precision. Relevance is binary (nonzero is relevant).
            rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
            mean_binary_average_precision(rs)
                0.78333333333333333
            rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
            mean_binary_average_precision(rs)
                0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_binary_precision(r) for r in rs])


def ndcg_at_k(r, k, method=0):
    """
    Score is normalized discounted cumulative gain (ndcg). Relevance is positive real values.
    Can use binary as the previous methods.
        Example from:  http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
            r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
            ndcg_at_k(r, 1)
                1.0
            r = [2, 1, 2, 0]
            ndcg_at_k(r, 4)
                0.9203032077642922
            ndcg_at_k(r, 4, method=1)
                0.96519546960144276
            ndcg_at_k([0], 1)
                0.0
            ndcg_at_k([1], 2)
                1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
