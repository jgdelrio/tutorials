import os
import time
import numpy as np
import pandas as pd
from recommenders import *


def test_content_rec():
    metadata = pd.read_csv("../Recommenders/data/raw/medatada.zip", low_memory=False)
    ratings = pd.read_csv("../Recommenders/data/raw/ratings_small.zip", low_memory=False)
    metadata = metadata[metadata.id.isin(ratings.movieId.unique())]

    content_rec = ContentRecommender()
    content_rec.fit_transform(metadata, 'id', 'overview')
    print(f"Training Time: {content_rec.fit_time:.4f}")

    content_recommendations = content_rec.recommend_df(ratings, 'userId', 'movieId')
    print(f"Rec calculated in: {content_rec.rec_time}!!")


def test_most_popular_catalogue():
    metadata = pd.read_csv(
        os.path.abspath(os.path.join(os.pardir, "Recommenders", "data", "processed", "metadata.zip")),
        low_memory=False)
    most_pop = MostPopular()
    most_pop.fit(metadata, 'id', vote_count='vote_count', vote_mean='vote_average')
    most_pop.predict_time
    most_pop.predict(rdn=True)


def test_most_popular_scores():
    ratings = pd.read_csv(
        os.path.abspath(os.path.join(os.pardir, "Recommenders", "data", "raw", "ratings_small.zip")),
        low_memory=False)
    most_pop = MostPopular()
    most_pop.fit(ratings, 'movieId', scores_field='rating', is_catalogue=False)
    most_pop.fit_time


if __name__ == '__main__':
    test_most_popular_catalogue()
