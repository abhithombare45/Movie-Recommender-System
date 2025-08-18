import numpy as np
import pandas as pd

movies_df = pd.read_csv("./../../data/raw/tmdb_5000_movies.csv")

movies_df.head(1)

credits_df = pd.read_csv("./../../data/raw/tmdb_5000_credits.csv")

credits_df.head(1)
movies_df.shape
credits_df.shape

df = movies_df.merge(credits_df, on="title")
df.head(1)

df.info()
df[
    [
        "id",
        "genres",
        "keywords",
        "original_language",
        "title",
        "overview",
        "popularity",
        "release_date",
        "spoken_languages",
        "cast",
        "crew",
        "status",
    ]
]
