import time
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, accuracy, SVD, NormalPredictor


def load_df_to_surprise_data(df, scale=(0, 1), columns=None):
    """
    Transforms a dataframe into the format expected by the surprise library
    :param df: pandas dataframe
    :param scale: scale of the scores
    :param columns: list of columns to use, sorted as 'userId', 'itemId' and 'score'
    :return: data formated for the surprise library
    """
    if columns is None:
        columns = ['userId', 'itemId', 'score']
    # Required reader with scale parameter
    reader = Reader(rating_scale=scale)

    # Sort the columns in the correct order if possible
    data = Dataset.load_from_df(df[columns], reader)

    return data


def get_cv_idxs(n, cv_idx=0, val_pct=0.2, seed=42):
    """ Get a list of index values for Validation set from a dataset

    Arguments:
        n : int, Total number of elements in the data set.
        cv_idx : int, starting index [idx_start = cv_idx*int(val_pct*n)]
        val_pct : (int, float), validation set percentage
        seed : seed value for RandomState

    Returns:
        list of indexes
    """
    np.random.seed(seed)
    n_val = int(val_pct * n)
    idx_start = cv_idx * n_val
    idxs = np.random.permutation(n)
    return idxs[idx_start:idx_start + n_val]


def clean_str(x, spaces=False):
    """Lower case and eliminate spaces"""
    if isinstance(x, list):
        if spaces:
            return [str.lower(i) for i in x]
        else:
            return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if string exists. If not, return empty string
        if isinstance(x, str):
            if spaces:
                return str.lower(x)
            else:
                return str.lower(x.replace(" ", ""))
        else:
            return ''


def get_topn(x, topn=5, field='name'):
    """
    Given a list of dictionaries, extract the selected field and
    returns the list top n elements or entire list, whichever is more.
    """
    if isinstance(x, list):
        names = [i[field] for i in x]
        # Check if more than n elements exist. If yes, return only first ones.
        # If no, return entire list.
        if len(names) > topn:
            names = names[:topn]
        return names

    # Return empty list in case of missing/malformed data
    return []


def set_kargs(obj, **kargs):
    """
    Provided an object and a dictionary or arguments,
    tries to set the specified arguments into the object"""
    for ar in kargs.keys():
        if ar in obj.__dict__:
            setattr(obj, ar, kargs[ar])
        else:
            raise KeyError('Unknown parameter')


def weighted_rating(df, mean_vote_field=None, vote_count_field=None,
                    id_field=None, quantile=0.90, weighted_field='w_score'):
    # calculate total mean vote across all items
    mean_vote = df[mean_vote_field].mean()
    # minimun amount of votes to be considered
    m = df[vote_count_field].quantile(quantile)

    def item_weighted_rating(x, m=m, mean_vote=mean_vote):
        # votes per item
        n_votes = x[vote_count_field]
        # average of votes per article
        vote_av = x[mean_vote_field]
        # Calculation based on the IMDB formula
        return (n_votes/(n_votes + m) * vote_av) + (m/(m + n_votes) * mean_vote)

    df[weighted_field] = df.apply(item_weighted_rating, axis=1)


class MostPopular:
    def __init__(self, **kargs):
        self.topn = 10
        self.samples = 100
        self.most_popular = None
        self.fit_time = None
        self.predict_time = None
        set_kargs(self, **kargs)

    def fit(self, df, id_field, scores_field=None,
            vote_count=None, vote_mean=None, is_catalogue=True, weighted=True):
        """
        Fit method to calculate the Most Popular items based on a catalogue or a list of scores
        :param df: Pandas DataFrame that can be a catalogue or a list of scores
        :param id_field:     field name of the IDs
        :param scores_field: field name of the scores
        :param vote_count:   field name of the vote count if a catalogue of items is provided
        :param vote_mean:    field name of the mean of the vote if a catalogue of items is provided
        :param is_catalogue: (boolean) True by default
        :param weighted: (boolean) True by default. Apply weights to the scores
        :return:
        """
        t0 = time.time()
        weighted_field = 'w_score'
        if is_catalogue:
            # Catalogue type. Expected to contain for each item
            # the vote_count and the vote_mean
            if vote_count is None or vote_mean is None:
                raise ValueError("'vote_count' and 'vote_mean' must be defined")
            catalogue = df

        else:
            # List of scores type. Proceed to calculate the vote count
            # and the vote mean
            if scores_field is None:
                raise ValueError("'scores_field' must be defined")
            grp_count = df[[id_field,
                            scores_field]].groupby(id_field)[scores_field].count().reset_index()
            grp_count.columns = [id_field, 'vote_count']
            grp_count['vote_mean'] = df[[id_field,
                                         scores_field]].groupby(id_field)[scores_field].mean().values
            vote_mean = 'vote_mean'
            vote_count = 'vote_count'
            catalogue = grp_count

        if weighted:
            weighted_rating(catalogue, mean_vote_field=vote_mean, vote_count_field=vote_count,
                            weighted_field=weighted_field)
            score_for_most_popular = weighted_field
        else:
            score_for_most_popular = vote_mean

        self.most_popular = catalogue.sort_values(
            score_for_most_popular,
            ascending=False)[[id_field, score_for_most_popular]][:self.samples]

        self.fit_time = time.time() - t0

    def predict(self, topn=None, rdn=False):
        if topn:
            self.topn = topn
        if rdn:
            sample = random.sample(range(self.samples), self.topn)
            return self.most_popular.iloc[sample, :]
        else:
            return self.most_popular[:self.topn]


class ContentRecommender():
    def __init__(self, **kargs):
        self.cosine_sim = None
        self.indices = None         # series with the item IDs as index and the original index as value
        self.report_scores = True
        self.topn = 10
        self.fit_time = None
        self.predict_time = None
        set_kargs(self, **kargs)

    def fit(self, df, id_field=None, desc_field=None):
        if df is None:
            raise ValueError("'df' must contain a valid Pandas DataFrame")
        if any([k is None for k in [id_field, desc_field]]):
            raise ValueError("Please provide the parameters 'id_field' and 'desc_field'")
        t0 = time.time()
        # Define a TF-IDF Vectorizer Object and remove all english stop words such as 'the', 'a'
        tfidf = TfidfVectorizer(stop_words='english')

        # Replace NaN with an empty string
        df = df[[id_field, desc_field]].copy()
        df[desc_field] = df[desc_field].fillna('')

        # Construct the required TF-IDF matrix by fitting and transforming the data
        tfidf_matrix = tfidf.fit_transform(df[desc_field])

        # Compute the cosine similarity matrix
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        del tfidf_matrix

        # Reverse map of indices and movie titles
        unique_ids, idx_unique = np.unique(df[id_field].values, axis=0, return_index=True)

        self.indices = pd.Series(idx_unique, index=unique_ids).drop_duplicates()
        self.fit_time = time.time() - t0

    def predict_user(self, item):
        """Takes in movie title as input and outputs most similar movies"""
        clean = True

        if not isinstance(item, (list, np.ndarray)):
            item = [item]
            clean = False

        index_list = np.empty(item.size, dtype=np.int16, order='C')
        index_list.fill(-1)

        for idx, t in enumerate(item):
            # Get the index of the movie/s that matches the title/item
            if t in self.indices.index:
                index_list[idx] = self.indices[t]
        # Eliminate -1 entries
        index_list = index_list[~np.isin(index_list, -1)]

        n = self.cosine_sim.shape[0]
        sim_index = np.empty((1, n * index_list.size), dtype=int, order='C')
        sim_index.fill(-1)
        sim_scores = np.empty((1, n * index_list.size), dtype=np.float, order='C')
        base = 0
        for p, idx in enumerate(index_list):
            # Get the pairwise similarity scores of all movies with that movie
            mov_inx = np.arange(0, n, 1)
            mov_list = self.cosine_sim[idx]
            mask = np.isin(mov_inx, index_list)
            if np.sum(mask) > 0:
                # eliminate items already in the user knowledge/seen
                mov_inx[mask] = -1
                mov_list[mask] = -1
            sim_index[0, base:base+n] = mov_inx
            sim_scores[0, base:base+n] = mov_list
            base = n

        if clean:
            # clean repetitions
            sim_index, index = np.unique(sim_index, axis=0, return_index=True)
            sim_scores = sim_scores[index]

        # Sort the movies based on the similarity scores
        sorting_idx = (-sim_scores).argsort()

        # Top n scores
        top_scores = sim_scores[0, sorting_idx][0, :self.topn]
        top_index = sim_index[0, sorting_idx][0, :self.topn]

        # Get the item original reference
        orig_ref = self.indices[self.indices.isin(top_index)]

        # Return the topn most similar movies (ref, score)
        if self.report_scores:
            return np.array([orig_ref.index, top_scores])
        else:
            return np.array(orig_ref.index)

    def predict(self, df, user_field, item_field, topn=None, report_scores=None):
        """
        Predict method. Provided a DataFrame with user IDs and itemIDs returns the suggestions
        of items that that particular user/s hasn't discovered yet.
        :param df:            Pandas DataFrame
        :param user_field:    user field name
        :param item_field:    item field name
        :param topn:          number of items to return (typically 10)
        :param report_scores: (boolean) if the output must contain scores for the items returned
        :return:              recommendations for each user
        """
        t0 = time.time()
        if topn is not None:
            self.topn = topn
        if report_scores is not None:
            self.report_scores = report_scores
            
        x_gr = df.groupby([user_field])[item_field].apply(lambda x: np.sort(np.array(x)))
        rec = x_gr.apply(self.predict_user)
        self.predict_time = time.time() - t0
        return rec


if __name__ == '__main__':
    content_rec = ContentRecommender(topn=20)
