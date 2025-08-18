import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process



df = pd.read_pickle("../../data/processed_data/processed_df.pkl")


cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(df["tags"]).toarray()

len(cv.get_feature_names_out())
#5000
cv.get_feature_names_out()
# JUST to check data
features = cv.get_feature_names_out().tolist()
print(features)   # prints all
# We can see many words have appeared with similar context like 
# actor is lot similar to actors & many more words..
# lets install nltk for working onit.
df.info()

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# Finally ready with tags column
df["tags"] = df["tags"].apply(stem)

# If user search similar pattern of movies name 
# but did not got whole name just first/last 2-3 words 
# of movies then it should fetch most popular movie
def find_best_match(movie_name):
    match = process.extractOne(movie_name, df["tags"].tolist())
    if match and match[1] >= 60:
        return match[0]
    return None


similarity = cosine_similarity(vectors)

def recommend(movie_name):
    # first resolve fuzzy / partial title using tags
    best_match = find_best_match(movie_name)
    if best_match:
        print(f"Best match found by tags: {best_match}")
        movie_idx = df[df["tags"] == best_match].index[0]
        dist = similarity[movie_idx]
        movie_list = sorted(list(enumerate(dist)), reverse=True, key=lambda x: x[1])[1:21]
    else:
        # fallback: try fuzzy matching against titles
        title_matches = process.extract(movie_name, df["title"].tolist(), limit=5)
        title_matches = [match for match in title_matches if match[1] >= 40]
        if title_matches:
            print(f"No good tag match found. Best title matches: {[m[0] for m in title_matches]}")
            movie_list = []
            for match_title, score in title_matches:
                idx = df[df["title"] == match_title].index[0]
                movie_list.append((idx, 1.0))  # score placeholder
        else:
            # fallback: recommend top 5 most similar by cosine similarity to any movie with close tag
            print("No good tag or title match found. Recommending top movies by similarity to closest tag match.")
            # Find closest tag match with lower threshold
            matches = process.extract(movie_name, df["tags"].tolist(), limit=1)
            if matches and matches[0][1] >= 30:
                movie_idx = df[df["tags"] == matches[0][0]].index[0]
                dist = similarity[movie_idx]
                movie_list = sorted(list(enumerate(dist)), reverse=True, key=lambda x: x[1])[1:6]
            else:
                # fallback: just top 5 popular movies
                movie_list = df.sort_values(by="popularity", ascending=False).head(5).index.to_list()
                movie_list = [(idx, 1.0) for idx in movie_list]

    recs = pd.DataFrame(
        [(idx, df.iloc[idx].title, df.iloc[idx].popularity, df.iloc[idx].release_date) 
         for idx, _ in movie_list],
        columns=["idx", "title", "popularity", "release_date"]
    )

    recs["release_date"] = pd.to_datetime(recs["release_date"], errors="coerce")
    recs["popularity_group"] = (recs["popularity"] // 100)

    # Sort by popularity_group first, then latest release date
    recs = recs.sort_values(by=["popularity_group", "release_date"], ascending=[False, False])

    return recs.head(5)[["title", "popularity", "release_date"]]

df.head(1)
recommend('Harry Potter')
recommend('Batman')
df.info()





df["popularity"].unique()
# release_date, spoken_languages
max:875
min 0.000372
