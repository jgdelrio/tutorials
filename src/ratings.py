# Rating prediction (error measures):
#   - MAE (Mean Absolute Error)
#   - MSE (Mean Squared Error)
#   - RMSE (Root mean Squared Error)

# Evaluation of recommendations (relevancy levels):
#   - Precision@n
#   - Recall
#   - Coverage (Diversity)
#   - nDCG: normalized Discounted Cummulative Gain
#   - MADCG: Mean Average Normalized Discounted Cummulative Gain
#   - MAP: Mean Average Precision
#   - Intra-list Similarity
#   - Lathia's Diversity
#   - Serendipity

# Implicit Feedback
#   - Mean Percentage Ranking
#   - User-Centric Evaluation Frameworks

import numpy as np
import pandas as pd
import math
from os.path import join
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import defaultdict
import matplotlib.pyplot as plt

from src import rank_metrics


# Some auxiliar functions
def number2string(n):
    fn = lambda x: x.to_bytes(math.ceil(x.bit_length() / 8), 'little').decode()
    if isinstance(n, (int, float)):
        return fn(n)
    elif isinstance(n, pd.core.series.Series):
        if isinstance(n[0], str):
            n = n.apply(int)
        return n.apply(fn)
    else:
        raise TypeError('Unexpected data type')


def merge_by(df, search_field, lookup_df, lookup_field, extra_fields=None, drop_duplicates=False):
    """
    Left merge into df of the info stored in lookup_df, as per the fields search_field and lookup_field respectively
    :param df:           pandas dataframe
    :param search_field: field name to use as index
    :param lookup_df:    pandas dataframe
    :param lookup_field: field name to use as index
    :param extra_fields: (optional) list of extra fields to be merged into df
    :return:
    """
    # Validate extra_fields parameter
    if extra_fields is None:
        extra_fields = lookup_df.columns.tolist()
    elif not isinstance(extra_fields, list):
        extra_fields = list(extra_fields)
    if lookup_field not in extra_fields:
        extra_fields.append(lookup_field)

    # Capture the items necessary to complete the df creating a lookup table
    lookup_items = df[search_field].unique().tolist()
    lookup = lookup_df[lookup_df[lookup_field].isin(lookup_items)].drop_duplicates(lookup_field).reset_index(drop=True)

    # Merge complementary dataframe...
    # df_merged = pd.merge(df, lookup[extra_fields],
    #                      left_on=search_field, right_on=lookup_field)
    df_merged = pd.merge(df, lookup[extra_fields],
                         how='left', left_on=search_field, right_on=lookup_field)

    if drop_duplicates:
        # Filter duplicates as per the search field...
        df_merged = df_merged.drop_duplicates(subset=search_field, keep='first')
    if lookup_field != search_field:
        df_merged.drop([lookup_field], axis=1, inplace=True)

    return df_merged


def align_metrics(predictions, ratings, keys=None):
    """
    Used to ensure that the error metrics are calculated over the same user-iid combinations
    Receive one dataframe for the predictions and one for the ratings data
    Both dataframes have the keys 'user', 'item' and 'rating' in that order

    :param predictions: dataset of predictions
    :param ratings:      validation dataset
    :param keys:        names of the keys that reference 'user', 'item' and 'rating' in that order
    :return:            merged dataframe of the user-item pairs included in both dataframes
    """
    if not isinstance(predictions, pd.core.frame.DataFrame):
        raise TypeError("predictions must be a pandas dataframe")
    if not isinstance(ratings, pd.core.frame.DataFrame):
        raise TypeError("ratings must be a pandas dataframe")

    if keys is None:
        keys = ['user', 'item', 'rating']

    ratings = ratings.iloc[:, :3].copy()
    predictions = predictions.iloc[:, :3].copy()

    ratings.columns = keys
    predictions.columns = [*keys[:2], 'predictions']

    # n_pred = len(predictions)
    # n_orig = len(ratings)

    # Drop users or items not included in the source data
    # predictions = predictions[predictions[keys[0]].isin(ratings[keys[0]])]
    # predictions = predictions[predictions[keys[1]].isin(ratings[keys[1]])]

    # # Drop users or items not included in the prediction
    # ratings = ratings[ratings[keys[0]].isin(predictions[keys[0]])]
    # ratings = ratings[ratings[keys[1]].isin(predictions[keys[1]])]

    # ratings['rating'] = ratings['rating'].astype(np.float64)
    # predictions['predictions'] = predictions['predictions'].astype(np.float64)

    # Detect missing values
    idx_missing_users = ~ratings[keys[0]].isin(predictions[keys[0]].tolist())
    idx_missing_items = ~ratings[keys[1]].isin(predictions[keys[1]].tolist())
    if any(idx_missing_users):
        print("\nMissing users in predictions:\n{}".format(set(ratings[keys[0]][idx_missing_users].values)))
    if any(idx_missing_items):
        print("\nMissing items in predictions:\n{}".format(set(ratings[keys[1]][idx_missing_items].values.tolist())))

    cmp_table = pd.merge(ratings,
                         predictions,
                         left_on=keys[:2],
                         right_on=keys[:2],
                         how='left')
    cmp_table.fillna(0, inplace=True)

    return cmp_table


def drop_repetitions(predictions, sourcedf, keys=None):
    """
    Receive one dataframe for the predictions and one for the test
    Both dataframes have the keys 'user', 'item' and 'rating' in that order

    :param predictions: dataset of predictions
    :param sourcedf:    dataset against which we verify existing user-item pairs
    :param keys:        names of the keys that reference 'user', 'item' and 'rating' in that order
    :return:            predictions without user-item pair values already in sourcedf
    """
    if not isinstance(predictions, pd.core.frame.DataFrame):
        raise TypeError("predictions must be a pandas dataframe")
    if not isinstance(sourcedf, pd.core.frame.DataFrame):
        raise TypeError("sourcedf must be a pandas dataframe")
    if keys is None:
        keys = ['user', 'item', 'rating']

    sourcedf = sourcedf.iloc[:, :3].copy()
    predictions = predictions.iloc[:, :3].copy()

    sourcedf.columns = keys
    predictions.columns = [*keys[:2], 'predictions']

    # Drop pairs of user-item already in the sourcedf
    i1 = predictions.set_index(keys[:2]).index
    i2 = sourcedf.set_index(keys[:2]).index
    recs = predictions[~i1.isin(i2)]
    if recs.shape[0] == 0:
        print('All the elements are already in the comparision dataset. The resulting filtered dataframe is empty!')
    return recs


def mae(predictions, testdf, keys=None, mlflow_logger=None):
    """
    Mean Absolute Error of the predictions vs the test set where the user-iid combinations are sorted automatically
    """
    if keys is None:
        keys = ['user', 'item', 'rating']
    cmp_table = align_metrics(predictions, testdf, keys)
    rst = mean_absolute_error(cmp_table[keys[2]], cmp_table['predictions'])
    if mlflow_logger is not None:
        mlflow_logger.log(rst, 'MAE', log_of='metrics')
    return rst


def mse(predictions, testdf, keys=None, mlflow_logger=None):
    """
    Mean Squared Error where the user-iid combinations are sorted automatically
    """
    if keys is None:
        keys = ['user', 'item', 'rating']
    cmp_table = align_metrics(predictions, testdf, keys)
    rst = mean_squared_error(cmp_table[keys[2]], cmp_table['predictions'])
    if mlflow_logger is not None:
        mlflow_logger.log(rst, 'MSE', log_of='metrics')
    return rst


def rmse(predictions, testdf, keys=None, mlflow_logger=None):
    """
    Root Mean Squared Error where the user-iid combinations are sorted automatically
    """
    if keys is None:
        keys = ['user', 'item', 'rating']
    cmp_table = align_metrics(predictions, testdf, keys)
    rst = np.sqrt(((cmp_table['predictions'] - cmp_table[keys[2]]) ** 2).mean())
    if mlflow_logger is not None:
        mlflow_logger.log(rst, 'RMSE', log_of='metrics')
    return rst


def error_measures(predictions, testdf, keys=None, output='dict', mlflow_logger=None):
    """
    MAE, MSE and RMSE where the user-iid combinations are sorted automatically
    """
    if keys is None:
        keys = ['user', 'item', 'rating']
    cmp_table = align_metrics(predictions, testdf, keys)

    mae_m = mean_absolute_error(cmp_table[keys[2]], cmp_table['predictions'])
    mse_m = mean_squared_error(cmp_table[keys[2]], cmp_table['predictions'])
    rmse_m = np.sqrt(((cmp_table['predictions'] - cmp_table[keys[2]]) ** 2).mean())

    rst_dict = {'mae': mae_m, 'mse': mse_m, 'rmse': rmse_m}

    if mlflow_logger is not None:
        mlflow_logger.log(rst_dict, rst_dict.keys(), log_of='metrics')

    if output == 'dict':
        return rst_dict
    else:
        return mae_m, mse_m, rmse_m


def precision_recall_at_k(predictions, testdf, keys=None, k=10, threshold=0.5,
                          mlflow_logger=None):
    """
    Precision: Proportion of recommended items in the top-k set that are relevant to the user
    Recall: Proportion of the total relevant items found in the top-k recommendations (likely to be low
            the greater the amount of relevant items as we limit the recommendations to k items)
    f1 score: Weighted average of precision and recall
    :param predictions:
    :param testdf:
    :param keys:          Typically to compare data vs predictions both have the first three columns sorted
                              as ['user', 'item', 'rating']
    :param k:             Sets how many results will be compared from the top results
    :param threshold:     Sets the point in which a prediction is considered relevant (ex: 0.6 in scale 0-1)
    :param mlflow_logger: mlflow_logger object to store the metrics in the corresponding run
    :return:
    """
    if keys is None:
        keys = ['user', 'item', 'rating']
    # Obtain the table with correlated ratings and predictions
    cmp_table = align_metrics(predictions, testdf, keys)

    # First map the list of users
    user_list = cmp_table[keys[0]].unique()

    precisions = dict()
    recalls = dict()
    for uid in user_list:
        user_ratings = cmp_table[cmp_table[keys[0]] == uid]

        # Sort user ratings by estimated value
        user_ratings = user_ratings.sort_values(by='predictions', axis=0, ascending=False)

        # Number of relevant items: keys[2] = 'rating'
        n_rel = user_ratings[keys[2]].loc[user_ratings[keys[2]] >= threshold].count()

        # Number of recommended items in top k
        n_rec_k = user_ratings['predictions'][:k].loc[user_ratings['predictions'] >= threshold].count()

        # Number of relevant and recommended items in top k
        top_k = user_ratings.reset_index()[:k]
        n_rel_and_rec_k = top_k[keys[0]].loc[
            (top_k[keys[2]] >= threshold) & (top_k['predictions'] >= threshold)].count()

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    avg_prec = sum(precisions.values()) / float(len(precisions.values()))
    avg_recall = sum(recalls.values()) / float(len(recalls.values()))
    avg_f1 = 2 * (avg_prec * avg_recall) / (avg_prec + avg_recall)

    if mlflow_logger is not None:
        at = 'At' + str(k) + 'tr' + str(threshold)
        mlflow_logger.log([avg_prec, avg_recall],
                          ['precision' + at, 'recall' + at],
                          log_of='metrics')

    return avg_prec, avg_recall, avg_f1, precisions, recalls


def get_top_n(predictions, sourcedf=None, n=10, userid=None, output='dataframe'):
    """
    Return the top-N recommendation for each user from a set of predictions.
    :param predictions: dataframe of predictions
    :param sourcedf:    source dataframe. Must be provided if we want only new results in the top 10 predictions.
                        If not provided, the top n may include results that were already in the source data
    :param n:           The number of recommendation to output for each user.
                        Default is 10.
    :param userid:      If specified, returns only the top_n for that user
    :param output:      It can be 'dict' or 'dataframe'
    :return:            Dataframe of users-item-rating
                        or dictionary of users with array of top_n recommendations as (iid-rating) tuples
    """

    # Initial validation
    if not isinstance(predictions, pd.core.frame.DataFrame):
        raise TypeError("predictions must be a pandas dataframe")
    if output not in ['dict', 'dataframe']:
        raise ValueError("The value of output must be 'dict' or 'dataframe'")

    if sourcedf is None:
        pred = predictions.copy()
    else:
        if not isinstance(sourcedf, pd.core.frame.DataFrame):
            raise TypeError("sourcedf must be a pandas dataframe")
        # No repetitions requested: filter out existing matches
        pred = drop_repetitions(predictions, sourcedf)

    predc = pred.columns

    # Filter users
    if userid is not None:
        pred = pred[pred.iloc[:, 0] == userid]

    pred = pred.reset_index(drop=True)

    if output is 'dict':
        top_n = defaultdict(list)
        for uid, iid, est in pred.values:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

    else:
        top_n_grp = pred.groupby(predc[0])[predc[2]].nlargest(n)
        top_n = pred.iloc[top_n_grp.index.get_level_values(1)]

    return top_n


def cmp_top_n(predictions, df, n=10, iid_ref=None, uids=None, iids=None, return_subsets=False):
    """
    Returns a comparison of the top 10 selection of a user vs the top 10 predictions
    :param predictions: Dataset of predictions in columns sorted as user-item-ratio
    :param df:          Original dataset in columns sorted as user-item-ratio
    :param n:           Number of elements to retrieve and compare
    :param iid_ref:     Reference table of iid to titles for example (optional)
                        or 'convert2string' if the iids were strings converted to numbers
    :param uids:        subset of the users to filter out from the datasets (optional)
    :param iids:        subset of the items to filter out from the datasets (optional)
    :param return_subsets: (boolean) if the subsets topn_data, topn_pred are desired
    :return:            Comparison
    """
    if not isinstance(predictions, pd.core.frame.DataFrame):
        raise TypeError("predictions must be a pandas dataframe")
    if not isinstance(df, pd.core.frame.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    if (uids is not None) and (not isinstance(uids, (pd.Series, list, np.ndarray))):
        raise TypeError("'uids' must be a list or Series")
    if (iids is not None) and (not isinstance(uids, (pd.Series, list, np.ndarray))):
        raise TypeError("'iids' must be a list or Series")

    # Get pred.count()keys whatever they are
    pkeys = predictions.columns
    dkeys = df.columns

    # In any case the users in df should be a subset of the users in predictions.
    pred = predictions.loc[predictions[pkeys[0]].isin(df[dkeys[0]].unique())]

    # Filter iids and uids as per iid_ref and uids respectively
    if (iids is not None) or (uids is not None):
        if (iids is not None) and (uids is not None):
            idx_pred = (pred[pkeys[0]].isin(uids)) & (pred[pkeys[1]].isin(iids))
            idx_data = (df[dkeys[0]].isin(uids)) & (df[dkeys[1]].isin(iids))
        elif iids is not None:
            idx_pred = pred[pkeys[1]].isin(iids)
            idx_data = df[dkeys[1]].isin(iids)
        elif uids is not None:
            idx_pred = pred[pkeys[0]].isin(uids)
            idx_data = df[dkeys[0]].isin(uids)

        # Apply filter
        pred = pred.loc[idx_pred]
        data = df.loc[idx_data]

    else:

        data = df

    # Get top n from data and predictions
    # The output are dictionaries of users with arrays of tuples(iid, rating)
    topn_data = get_top_n(data, n=n, output='dataframe')
    topn_pred = get_top_n(pred, sourcedf=data, n=n, output='dataframe')

    # Store the indexes as per ratings data
    # idx_topn_data = topn_data.index.values
    # idx_topn_pred = topn_pred.index.values

    # Merge dataframes: user / dataItemId / dataItemRating / predictionItemId /  predictionItemRating
    if topn_data.shape[0] == topn_pred.shape[0]:
        topn_data.reset_index(inplace=True, drop=True)
        print(topn_data.columns)
        print(['data_user', 'data_iid', 'data_rating', *['data_'+k for k in dkeys[3:]]])
        topn_data.columns = ['data_user', 'data_iid', 'data_rating', *['data_'+k for k in dkeys[3:]]]

        topn_pred.reset_index(inplace=True, drop=True)
        topn_pred.columns = ['pred_user', 'pred_iid', 'pred_rating', *['pred_'+k for k in pkeys[3:]]]

        topn_cmp = pd.merge(topn_data, topn_pred, left_index=True, right_index=True)
        # Eliminate repeated user id
        topn_cmp.drop(['pred_user'], axis=1, inplace=True)

    else:
        raise TypeError("The size of the datasets don't match. "
                        "Please make sure the source data has enough results for each user")

    # Apply the iid_ref if provided
    try:
        if iid_ref is not None:
            if iid_ref == 'convert2string':
                columns = ['data_iid', 'pred_iid']
                topn_cmp[columns] = topn_cmp[columns].apply(lambda x: number2string(x))
            else:
                pass
    except Exception as err:
        print('Exception while converting iid to string: {}'.format(err))

    # Return comparison dataset
    if return_subsets is True:
        return topn_cmp, topn_data, topn_pred
    else:
        return topn_cmp


def get_per_fields(df_base, fields_extract, field_search, val_search):
    """
    Receive a df_base of which we want to extract to the rows and columns as per val_search and field_search
    :param df_base:
    :param fields_extract: Fields that will be captured from df_base
    :param field_search: Fields used for the search
    :param val_search: Fields used for the search
    :return:
    """
    # Initial validation
    if not isinstance(df_base, pd.core.frame.DataFrame):
        raise TypeError("df_base must be a pandas dataframe")
    if not isinstance(field_search, str):
        raise TypeError('field_search must be a string')
    if isinstance(val_search, str):
        val_search = [val_search]
    if not (field_search in df_base.columns.values):
        raise KeyError('Field not found in dataset')

    idx = df_base[field_search].isin(val_search)
    df = df_base[[idx, *fields_extract]].drop_duplicates(idx, keep='first')

    return df


class RatingsAccum(object):
    """
    Accumulator or metrics
    """
    def __init__(self, metrics=('MAE', 'MSE', 'RMSE', 'Precision', 'Recall')):
        # Init x axis references
        self.X = []
        self.metric_names = metrics
        self.metrics = {}
        for m in metrics:
            self.metrics[m] = []

    def add_metrics(self, x, metrics):
        is_list = False
        if isinstance(x, list):
            self.X.extend(x)
            is_list = True
        elif isinstance(x, (int, float)):
            self.X.append(x)
        else:
            raise TypeError('Unexpected type: Nor int nor float')

        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if is_list:
                    self.metrics[k].extend(v)
                else:
                    self.metrics[k].append(v)

    @staticmethod
    def setax(x, y, ax, y_label, title=None):
        ax.plot(x, y, 'b-')
        ax.margins(x=0, y=0)
        if title is not None:
            ax.set_title(title)
        ax.set_ylabel(y_label, color='b')
        ax.tick_params('y', colors='b')

    def plot(self, metric_name):
        fct_min = 0.9
        fct_max = 1.1

        # fig, ax = plt.subplots()
        fig = plt.figure()

        if metric_name in self.metrics.keys():
            m = self.metrics[metric_name]
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8],
                              xticklabels=[], ylim=(-1.2, 1.2))
            self.setax(self.X, m, ax, 'Model Metrics: ' + metric_name, metric_name)

        elif metric_name == 'all':
            n_metrics = len(self.metrics.keys())
            section = 0.8 / n_metrics
            base = 0

            for n, mk in enumerate(self.metrics.keys()):
                m = self.metrics[mk]
                miny = fct_min * min(m)
                maxy = fct_max * max(m)
                if n == 0:
                    ax = fig.add_axes([0.1, base, 0.8, section],
                                      ylim=(miny, maxy))
                    self.setax(self.X, m, ax, 'Model Metrics: ' + mk, mk)

                else:
                    ax = fig.add_axes([0.1, base, 0.8, section],
                                      xticklabels=[], ylim=(miny, maxy))
                    self.setax(self.X, m, ax, 'Model Metrics: ' + mk, mk)
                base += section

        plt.show()


def sparcity(df, ismatrix=False, index_col=None, category_col=None, val_col=None):
    """
    Calculate the sparcity of a given df (ex: items in columns, users in rows and values filling the df)
    :param df:           Dataframe with a range of values. If a dataframe is defines with index, categories
                         and values in three different columns, then ismatrix must be defined as False
    :param ismatrix:     Bollean that defines the behaviour for the calculation
    :param index_col:    String that defines the index column
    :param category_col: String that defines the categories column
    :param val_col:      String that defines the values column
    :return:             Sparcity value
    """
    if not isinstance(df, pd.core.frame.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    if ismatrix:
        # Calculate sparcity
        sparcity_val = float(len(df.values.nonzero()[0]))
        sparcity_val /= (df.shape[0] * df.shape[1])
    else:
        if (index_col is None) or (category_col is None) or (val_col is None):
            raise ValueError('index_col, category_col and val_col are required')
        sparcity_val = float(df[val_col].astype(bool).sum(axis=0))
        sparcity_val /= (df[index_col].nunique() * df[category_col].nunique())
        # df = matrix[[index_col, category_col, val_col]]
        # matrix = df.pivot_table(index=index_col, columns=category_col, values=val_col).fillna(0)

    sparcity_val *= 100
    return round(sparcity_val, 2)


def catalogue_coverage(catalogue, recommended, 
                       cfield='iid', rfield='iid',
                       r_rating_field=None, rating_threshold=0.5,
                       mlflow_logger=None, label=None):
    """
    Degree to which recommendations cover the set of available elements and the degree
    to which recommendations can be generated to all potential users.
    It represents a more broad investigation of the product space.

    Ex: Number of unique recommended items vs the total number of unique items
        Other coverage evaluations... by year, by certificate, by genres, etc.
    
    :param catalogue:     dataframe of the full catalogue
    :param recommended:   dataframe of recommendations
    :param cfield:        field name in the catalogue (can be iid, year, cert, etc depending on the evaluation)
    :param rfield:        field name in the recommendations
    :param r_rating_field:    (optional) field name of the Ratings in the recommendations
    :param rating_threshold:  (optional) threshold to apply in the recommendations for filtering
    :param mlflow_logger: if provided it will sent the metrics to the experiment and session stored in it
    :param label:         Used in mlflow as tag for the metric
    :return:                  catalogue_coverage value
    """
    try:
        n_available_items = catalogue[cfield].nunique()
    except TypeError:
        # If elements within the field contains a list...
        is_na = recommended[rfield].isna()
        n_available_items = len(set().union(*catalogue[~is_na][cfield]))
        if any(is_na):
            warnings.warn("There are {} null values in the catalogue '{}' field".format(sum(is_na), cfield))
    
    # Apply threshold is provided
    if r_rating_field is not None:
        recommended = recommended[recommended[r_rating_field] >= rating_threshold]

    if rfield not in recommended.columns:
        # The predictions may not have the parameter (ex: year) but it should be in the catalogue
        # and can be extracted with a merge by item search
        if rfield not in catalogue.columns:
            raise Exception('rfield ({}) not found'.format(rfield))
        recommended = merge_by(recommended, 'iid', catalogue, 'iid', [cfield])

    try:
        n_recommended_items = recommended[rfield].nunique()
    except TypeError:       # If elements within the field contains a list...
        is_na = recommended[rfield].isna()
        n_recommended_items = len(set().union(*recommended[~is_na][rfield]))
        if any(is_na):
            warnings.warn("There are {} null values in the recommendations '{}' field".format(sum(is_na), rfield))

    cat_coverage = n_recommended_items / n_available_items

    if mlflow_logger is not None:
        if label is None:
            label = 'coverage'
        mlflow_logger.log(cat_coverage, label, log_of='metrics')

    return cat_coverage


def catalogue_coverage_distr(catalogue, recommended, cfield='iid', rfield='iid',
                             r_rating_field=None, rating_threshold=0.5,
                             mlflow_logger=None, label=None, root='.'):
    """
    Generates the distribution of elements within the catalogue and the recommendations for the fields specified
    :param catalogue:      dataframe of the full catalogue
    :param recommended:    dataframe of recommendations
    :param cfield:         fieldname to evaluate in catalogue
    :param rfield:         fieldname to evaluate in recommendations
    :param r_rating_field: rating fieldname in recommendations
    :param rating_threshold: threshold to apply in recommendations rating field
    :param mlflow_logger:  mlflow_logger object
    :param label: l        label used to store the artifact (image) in mlflow

    :return: catalogue_coverage value
    """
    # Apply threshold is provided
    if r_rating_field is not None:
        recommended = recommended[recommended[r_rating_field] >= rating_threshold]

    try:
        catalogue_dist = catalogue.groupby(cfield).count().iloc[:, 0].reset_index()
    except TypeError as err:
        if err.args[0] != "unhashable type: 'list'":
            raise Exception('Unexpected exception. Please update with a new exception handling')
        is_na = catalogue[cfield].isna()
        if any(is_na):
            warnings.warn("There are {} null values in the catalogue '{}' field".format(sum(is_na), cfield))
        np_cat = catalogue[~is_na][cfield].values
        cat_count = {}
        for i in np_cat:
            for e in i:
                try:
                    cat_count[e] += 1
                except KeyError:
                    cat_count[e] = 1

        # np.split(np_cat[:, 1], np.cumsum(np.unique(np_cat[:, 0], return_counts=True)[1][:-1]))

        catalogue_dist = pd.DataFrame.from_dict(cat_count, orient='index', columns={'count'}).sort_index().reset_index()
    catalogue_dist.columns = [cfield, 'count']

    try:
        recommended_dist = recommended.groupby(rfield).count().iloc[:, 0].reset_index()
    except TypeError:
        is_na = recommended[rfield].isna()
        if any(is_na):
            warnings.warn("There are {} null values in the recommendations '{}' field".format(sum(is_na), rfield))
        np_rec = recommended[~is_na][rfield].values
        rec_count = {}

        for i in np_rec:
            for e in i:
                try:
                    rec_count[e] += 1
                except KeyError:
                    rec_count[e] = 1        

        recommended_dist = pd.DataFrame.from_dict(
            rec_count, orient='index', columns={'count'}).sort_index().reset_index()
    recommended_dist.columns = [rfield, 'count']

    if mlflow_logger is not None:
        # Logging into MLflow
        if label is None:
            label = 'coverage_distrib'
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(catalogue_dist[cfield], catalogue_dist['count'], marker='x', color='b')
        ax2.plot(recommended_dist[rfield], recommended_dist['count'], marker='x', color='r')

        for tick in [*ax1.get_xticklabels(), *ax2.get_xticklabels()]:
            tick.set_rotation(90)

        ax1.set_title(label + ' Cat.')
        ax2.set_title(label + ' Rec.')
        fig_name = join(root, 'reports', 'figures', (label + '.png'))

        fig.savefig(fig_name)
        mlflow_logger.log_artifact(fig_name)
        plt.close(fig)

    return catalogue_dist, recommended_dist


def item_score_ndcg(predicted_scores, user_scores, k=10, method=0, idcg=False, mlflow_logger=None,
                    iid_user=None, iid_pred=None,
                    rtg_user=None, rtg_pred=None):
    if iid_user is None:
        iid_user = user_scores.columns[0]
    if iid_pred is None:
        iid_pred = predicted_scores.columns[0]
    if rtg_user is None:
        rtg_user = user_scores.columns[1]
    if rtg_pred is None:
        rtg_pred = predicted_scores.columns[1]

    # Sort user scores
    user_scores = user_scores.sort_values([rtg_user], ascending=False)

    # Sort predictions as per user_scores filling up missing entries with 0's
    idx = rank_metrics.get_index(predicted_scores[iid_pred].values, user_scores[iid_user].values)
    idx_bool = [x is not None for x in idx]
    idx_relevant = [i for i, r in zip(idx, idx_bool) if r is True]
    predicted_scores = predicted_scores[idx_bool]
    pred_scores = user_scores.copy()
    # Recreate the predictions based of user_scores items with predictions values
    pred_scores[rtg_user] = 0.
    for i, e in zip(idx_relevant, predicted_scores[rtg_pred].values):
        pred_scores.loc[i, rtg_user] = e

    # Calculate ndcg
    ndcg = rank_metrics.prd_ndcg_at_k(pred_scores[rtg_user].values, user_scores[rtg_user].values,
                                      k=k, method=method, idcg=idcg)
    if mlflow_logger is not None:
        # Send metrics
        mlflow_logger.log(ndcg, 'ndcg', log_of='metrics')
    return ndcg


def mandcg_at_k(predicted_scores, user_scores, k=10, method=0, idcg=False, mlflow_logger=None,
                uid_user=None, uid_pred=None, iid_user=None, iid_pred=None, rtg_user=None, rtg_pred=None):
    """
    Mean Average Normalized Discounted Cumulative Gain (mean between users)

    For every user we find ndcg_at_k and then calculate the mean and the standard deviation
    :param predicted_scores: df of predicted scores containing user, items and ratings
    :param user_scores:      df of user scores containing user, items and ratings
    :param k:                level at which mandcg is calculated
    :param method:
    :param idcg:             apply ideal dcg in the calculation
    :param mlflow_logger:    mlflow_logger object
    :return: mean and standard deviation
    """

    if uid_user is None:
        uid_user = user_scores.columns[0]
    if uid_pred is None:
        uid_pred = predicted_scores.columns[0]

    # Get list of users from the user_scores
    user_list = user_scores[uid_user].unique()

    def get_ndcg(user):
        pr_sample_user = predicted_scores[predicted_scores[uid_pred] == user]

        # Calculate ndcg
        return item_score_ndcg(pr_sample_user.drop([uid_pred], axis=1),
                               user_scores[user_scores[uid_user] == user].drop([uid_user], axis=1),
                               k=k, method=method, idcg=idcg,
                               iid_user=iid_user, iid_pred=iid_pred, rtg_user=rtg_user, rtg_pred=rtg_pred)

    ndcg_list = list(map(get_ndcg, user_list))
    mandcg = np.mean(ndcg_list)

    if mlflow_logger is not None:
        # Send metrics
        mlflow_logger.log(mandcg, 'mandcg', log_of='metrics')

    return mandcg, np.std(ndcg_list)


def catalogue_coverage_by_user(catalogue, recommended,
                               cuid='uid', cfield='iid',
                               ruid='uid', rfield='iid'):
    """
    Similar to catalogue_coverage but per user
    :param catalogue:
    :param recommended:
    :param cuid:        field name of the catalogue user Id
    :param cfield:
    :param ruid:        field name of the recommended user Id
    :param rfield:
    :return:
    """
    # For every user group the ratings by genre
    cat_grp = catalogue.groupby([cuid, cfield])
    rec_grp = recommended.groupby([ruid, rfield])

    # For every user...
    # 1- ratings of intersection of recommended genres with the user genres
    # 2- ratings of non intersecting elements between recommended genres and watched genres

    # TBC ####################################
    return 1


# def weighted_catalogue_coverage(recommendation, catalogue, riid='iid', ciid='iid'):
#     # Similar to catalogue coverage but consider only relevant items for the recommended as well as for the catalogue
#     n_recommended_items = recommendation[riid].nunique()
#     n_available_items = catalogue[ciid].nunique()
#     cat_coverage = n_recommended_items / n_available_items
#     return cat_coverage


def serendipity():
    # Novelty of recommendations and how far they may positively surprise users.
    # Showing something that the user didn't know existed but in which he is interested.

    # Given those parameters, we define serendipity as the unexpected recommendations (recommendations not
    # included in a primitive prediction model) that are useful for the user
    pass
