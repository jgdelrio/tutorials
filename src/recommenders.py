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


class ContentRecommender():
    def __init__(self):
        self.cosine_sim = None
        self.indices = None
        self.topn = 10

    def train(self, df, id_field, desc_field):
        # Define a TF-IDF Vectorizer Object and remove all english stop words such as 'the', 'a'
        tfidf = TfidfVectorizer(stop_words='english')

        # Replace NaN with an empty string
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

        if isinstance(item, str):
            item = [item]
            clean = False
        sim_scores = list()

        index_list = []
        for t in item:
            # Get the index of the movie that matches the title
            index_list.append(self.indices[t])

        for idx in index_list:
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