import pickle
from flask import Flask, request, render_template, jsonify
import datetime
import importlib.util
import os
import pandas as pd

app = Flask(__name__)

# -------------------
# Import recommend() from 03_model_bulding.py dynamically
# -------------------
module_path = os.path.join(
    os.path.dirname(__file__), "./src/Model_Building/02_data_featuring.py"
)
spec = importlib.util.spec_from_file_location("model_building", module_path)
model_building = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_building)
model_recommend = model_building.recommend  # <-- use this inside app
# -------------------


import requests  # NEW
import os
API_KEY = os.getenv("TMDB_API_KEY")
# TMDB_API_KEY = "c368f96d723aaf0d7eeb47f79d8c9d91"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"


# -------------------
# Understand this:
# -------------------
def tmdb_search(title, year=None):
    """Query TMDB for a title (optionally constrained by year)."""
    params = {"api_key": TMDB_API_KEY, "query": title}
    if year:
        # TMDB supports both 'year' and 'primary_release_year'; try the stricter one
        params["primary_release_year"] = year
    try:
        r = requests.get(TMDB_SEARCH_URL, params=params, timeout=8)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception:
        return []


def pick_with_poster(results):
    """Prefer a result that has a poster; fallback to first."""
    if not results:
        return None
    with_poster = [m for m in results if m.get("poster_path")]
    return with_poster[0] if with_poster else results[0]


def fetch_tmdb_details(title, year=None):
    """Return a dict: {title, poster, year, overview, tmdb_id}."""
    results = tmdb_search(title, year)
    best = pick_with_poster(results)
    if not best:
        return {
            "title": title,
            "poster": None,
            "year": year or "—",
            "overview": "No overview available.",
            "tmdb_id": None,
        }
    poster_path = best.get("poster_path")
    poster_url = f"{TMDB_IMG_BASE}{poster_path}" if poster_path else None
    release_date = (best.get("release_date") or "")[:10]
    yr = (release_date.split("-")[0] if release_date else year) or "—"
    return {
        "title": best.get("title") or title,
        "poster": poster_url,
        "year": yr,
        "overview": best.get("overview") or "No overview available.",
        "tmdb_id": best.get("id"),
    }


def enrich_with_tmdb(recs_df):
    """Take the DataFrame from model_recommend and add TMDB fields."""
    enriched = []
    if recs_df is None or recs_df.empty:
        return enriched
    for _, row in recs_df.iterrows():
        # Try to pass the year we already know for better matching
        year = None
        if "release_date" in recs_df.columns and pd.notna(row["release_date"]):
            # row["release_date"] might be a string or Timestamp
            s = str(row["release_date"])
            year = s[:4] if len(s) >= 4 else None
        enriched.append(fetch_tmdb_details(row["title"], year))
    return enriched


# -------------------
# -------------------


# Load pickle (dict with df + similarity)
with open("models/movie_recommender.pkl", "rb") as f:
    data = pickle.load(f)

movies = data["movies"]  # pandas DataFrame
similarity = data["similarity"]  # cosine similarity matrix


# -------------------
# Routes
# -------------------
@app.get("/")
def home():
    trending_movies = movies.sample(8)  # random trending, or .head(8)
    trending_results = enrich_with_tmdb(trending_movies)

    return render_template(
        "index.html",
        trending=trending_results,
        current_year=datetime.datetime.now().year,
        results=None,
        query="",
    )


@app.get("/suggest")
def suggest():
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])
    matches = [t for t in movies["title"] if q in t.lower()][:8]
    return jsonify(matches)


@app.post("/recommend")
def recommend():
    query = request.form.get("query", "").strip()
    # Use the new fuzzy-aware model_recommend
    recs_df = model_recommend(query)  # returns DataFrame
    results = enrich_with_tmdb(recs_df)  # list of dict title/poster/year/overview
    recs = recs_df["title"].tolist() if not recs_df.empty else []

    return render_template(
        "index.html",
        trending=list(movies["title"].head(8)),
        current_year=datetime.datetime.now().year,
        results=results,  # <— now list of dicts (not just strings)
        query=query,
    )


@app.get("/about")
def about():
    return render_template("about.html", current_year=datetime.datetime.now().year)


if __name__ == "__main__":
    app.run(debug=True)
