import numpy as np
import pandas as pd
import ast

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

df.drop(columns=["original_language"], inplace=True)
df.drop(columns=["spoken_languages"], inplace=True)

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

ast.literal_eval(
    '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
)


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
df["tags"] = df["tags"].apply(lambda x:" ".join(x))
df["tags"][0]


df = pd.to_pickle("../../data/processed_data/processed_df.pkl")











cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(df["tags"]).toarray()

len(cv.get_feature_names_out())
#5000
cv.get_feature_names_out()
# lets check out data
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
