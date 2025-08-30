import pandas as pd
import numpy as np
import sklearn
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
import pandas as pd
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
import flask
import os


print("numpy", np.__version__)
print("pandas", pd.__version__)
print("scikit-learn", sklearn.__version__)
# numpy 2.3.2
# pandas 2.3.1
# scikit-learn 1.7.1

movies_df = pd.read_csv("./../../data/raw/tmdb_5000_movies.csv")

movies_df.head(1)

credits_df = pd.read_csv("./../../data/raw/tmdb_5000_credits.csv")

credits_df.head(1)
movies_df.shape
credits_df.shape

df = movies_df.merge(credits_df, on="title")
# df_test = movies_df.merge(credits_df[["cast", "crew"]], on="title")
df.head(1)

df.info()
mdict = df
mdict.shape

df = df[
    [
        "id",
        "genres",
        "keywords",
        "title",
        "overview",
        "popularity",
        "release_date",
        "cast",
        "crew",
    ]
]

# df.drop(columns=["original_language"], inplace=True)
# df.drop(columns=["spoken_languages"], inplace=True)

df.shape
df.head(1)
df.info()

df.isnull().sum()
df[df["overview"].isna()]
df[df["release_date"].isna()]
# id:overview 370980, 459488, 292539
# id:release_date 380097

# Removing rows with NaN values
df.dropna(inplace=True)

df.duplicated().sum()


## Sorting Column data

df.iloc[0].genres
# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'

# to work on above output we need
# to convert this (JSON) output to list format.

# ast.literal_eval(
#     '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# )


# So modified function accordingly,
def convert(obj):
    g_list = []
    for i in ast.literal_eval(obj):
        g_list.append(i["name"])
    return g_list


# genres colunm conversation
df["genres"] = df["genres"].apply(convert)


# keywords column converstaion
df.iloc[0].keywords
df["keywords"] = df["keywords"].apply(convert)


# cast column conversation
df["cast"][0]


# we are taking 4 main actor names in consideration
# so lets create a function to catch names
def fetch_lead_actor(obj):
    a_list = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 4:
            a_list.append(i["name"])
            counter + 1
        else:
            break
    return a_list


df["cast"] = df["cast"].apply(fetch_lead_actor)

# crew column conversation
# We are fetching 3 main names from crew.
# Writer, Director & Producer (WDP)
df["crew"][0]


def fetch_WDP(obj):
    n_list = []
    for i in ast.literal_eval(obj):
        if i["job"] in ["Writer", "Director", "Producer"]:
            n_list.append(i["name"])
    return n_list


df["crew"] = df["crew"].apply(fetch_WDP)

# overview column formating
# column contains a string, we are converting/splitting
# it into list.

df["overview"] = df["overview"].apply(lambda x: x.split())


# Now we are removing Space betn the words
# so we can extracte features correctly without
# missmatching similar NAME person.
df["genres"] = df["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
df["keywords"] = df["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
df["cast"] = df["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
df["crew"] = df["crew"].apply(lambda x: [i.replace(" ", "") for i in x])
df.head(2)

# Now lets create new column that have
# all the coulmn data into single one.

df["tags"] = df["genres"] + df["keywords"] + df["cast"] + df["crew"] + df["overview"]
df.head(1)
# converting tags columns from list to string
df["tags"] = df["tags"].apply(lambda x: " ".join(x))
df["tags"][0]


df.to_pickle("../../data/processed_data/01_partially_featured_data.pkl")
