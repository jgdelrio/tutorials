import numpy as np
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


