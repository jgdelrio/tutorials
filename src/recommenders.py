import time
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


class ContentRecommender():
    def __init__(self, **kargs):
        self.cosine_sim = None
        self.indices = None         # series with the item IDs as index and the original index as value
        self.report_scores = True
        self.topn = 10
        self.train_time = None
        self.rec_time = None

        for ar in kargs.keys():
            if ar in self.__dict__:
                setattr(self, ar, kargs[ar])
            else:
                raise KeyError('Unknown parameter')

    def train(self, df, id_field, desc_field):
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
        self.train_time = time.time() - t0

    def get_recommendations(self, item, topn=None, report_scores=None):
        """Takes in movie title as input and outputs most similar movies"""
        clean = True
        if topn is not None:
            self.topn = topn
        if report_scores is not None:
            self.report_scores = report_scores

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

    def recommend_df(self, df_scores, user_field, item_field, topn=None):
        t0 = time.time()
        if topn is not None:
            self.topn = topn
        x_gr = df_scores.groupby([user_field])[item_field].apply(lambda x: np.sort(np.array(x)))
        rec = x_gr.apply(self.get_recommendations)
        self.rec_time = time.time() - t0
        return rec


if __name__ == '__main__':
    content_rec = ContentRecommender(topn=20)
