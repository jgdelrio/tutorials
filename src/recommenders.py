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
        self.indices = None
        self.topn = 10
        for ar in kargs.keys():
            if ar in self.__dict__:
                setattr(self, ar, kargs[ar])
            else:
                raise KeyError('Unknown parameter')

    def train(self, df, id_field, desc_field):
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
        self.indices = pd.Series(df.index, index=df[id_field]).drop_duplicates()

    def get_recommendations(self, item, topn=None):
        """Takes in movie title as input and outputs most similar movies"""
        clean = True
        if topn is None:
            topn = self.topn

        if not isinstance(item, list):
            item = [item]
            clean = False
        sim_scores = list()

        index_list = []
        for t in item:
            # Get the index of the movie that matches the title/item
            if t in self.indices:
                index_list.append(self.indices[t])
            else:
                index_list.append(None)

        for idx in index_list:
            if idx is not None:
                # Get the pairwise similarity scores of all movies with that movie
                mov_list = list(enumerate(self.cosine_sim[idx]))
                mov_list = [e for e in mov_list if e[1] not in index_list]    # Ignore itself
                sim_scores.extend(mov_list)

        # clean repetitions
        if clean:
            ref = [x[1] for x in sim_scores]
            unique_ref = []
            n = len(ref)
            for idx, r in enumerate(ref[::-1]):
                if r not in unique_ref:
                    unique_ref.append(r)
                else:
                    sim_scores.pop(n - idx - 1)

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[:(topn + 1)]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the topn most similar movies
        return self.indices[self.indices.isin(movie_indices)].index.tolist()

    def recommend_df(self, df_scores, user_field, item_field, topn=None):
        if topn is not None:
            self.topn = topn
        x_gr = df_scores.groupby([user_field])[item_field].apply(lambda x: sorted(list(x)))
        rec = pd.DataFrame({'user': x_gr.index.tolist()})
        rec['recommedation'] = x_gr.apply(self.get_recommendations)
        return rec


if __name__ == '__main__':
    content_rec = ContentRecommender(topn=20)
    pass
