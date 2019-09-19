import time
import numpy as np
import pandas as pd
from recommenders import *


def test_content_rec():
    metadata = pd.read_csv("../Recommenders/data/raw/medatada.zip", low_memory=False)
    ratings = pd.read_csv("../Recommenders/data/raw/ratings_small.zip", low_memory=False)
    metadata = metadata[metadata.id.isin(ratings.movieId.unique())]

    content_rec = ContentRecommender()
    content_rec.train(metadata, 'id', 'overview')
    print(f"Training Time: {content_rec.train_time:.4f}")

    content_recommendations = content_rec.recommend_df(ratings, 'userId', 'movieId')
    print(f"Rec calculated in: {content_rec.rec_time}!!")


if __name__ == '__main__':
    test_content_rec()
