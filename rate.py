# rate_app.py  – Movie Rating Prediction + Posters (Streamlit)

import ast
import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 0. TMDB API KEY (IMPORTANT)
# -----------------------------
# 👇 REPLACE this with your real TMDB "API Key (v3 auth)"
TMDB_API_KEY = "b84298fc0f62ad211030847c042d8717"


# -----------------------------
# 1. Helper functions
# -----------------------------
def parse_list_field(obj):
    """Parse JSON-like list-of-dicts string and return list of 'name' values."""
    if pd.isna(obj):
        return []
    if isinstance(obj, list):
        return [d.get("name", "") for d in obj if isinstance(d, dict)]
    try:
        data = ast.literal_eval(obj)
        if isinstance(data, list):
            return [d.get("name", "") for d in data if isinstance(d, dict)]
    except Exception:
        return []
    return []


def parse_cast_first(obj):
    names = parse_list_field(obj)
    return names[0] if names else "Unknown"


def parse_genre_first(obj):
    names = parse_list_field(obj)
    return names[0] if names else "Unknown"


def get_director(crew_obj):
    if pd.isna(crew_obj):
        return "Unknown"
    try:
        crew = ast.literal_eval(crew_obj)
        for person in crew:
            if isinstance(person, dict) and person.get("job") == "Director":
                return person.get("name", "Unknown")
    except Exception:
        pass
    return "Unknown"


def fetch_poster(tmdb_id):
    """
    Fetch poster URL from TMDB API using numeric movie id from the dataset.
    Returns None if key not set or poster not found.
    """
    if TMDB_API_KEY.startswith("PASTE_") or not TMDB_API_KEY:
        # user has not configured key yet
        return None

    if pd.isna(tmdb_id):
        return None

    try:
        tmdb_id = int(tmdb_id)
    except Exception:
        return None

    url = (
        f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        f"?api_key={TMDB_API_KEY}&language=en-US"
    )
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
    except Exception as e:
        # For debugging you can print(e) in the console
        print("Poster fetch failed:", e)
    return None


# -----------------------------
# 2. Load & preprocess data
# -----------------------------
@st.cache_data(show_spinner=True)
def load_and_engineer():
    st.write("Loading TMDB data and engineering features...")

    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits, on="id", how="left")

    # keep the movie title from movies table
    df["title"] = df["title_x"]

    # numeric
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")

    df["budget_log"] = np.log1p(df["budget"])

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    # main genre / actor / director (same logic as prediction_model.py)
    df["main_genre"] = df["genres"].apply(parse_genre_first)
    df["main_actor"] = df["cast"].apply(parse_cast_first)
    df["director_name"] = df["crew"].apply(get_director)

    # drop missing target
    df = df[~df["vote_average"].isna()].copy()

    numeric_features = [
        "popularity",
        "runtime",
        "budget_log",
        "vote_count",
        "release_year",
    ]
    categorical_features = [
        "main_genre",
        "main_actor",
        "director_name",
    ]

    X = df[numeric_features + categorical_features]
    y = df["vote_average"]

    return df, X, y, numeric_features, categorical_features


# -----------------------------
# 3. Build & train model
# -----------------------------
@st.cache_resource(show_spinner=True)
def train_model(X, y, numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    # metrics (RMSE computed without 'squared' arg to be sklearn-version-safe)
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    return pipe, metrics


# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Movie Rating Prediction", layout="wide")

st.title("🎬 Movie Rating Prediction")
st.write(
    """
This app uses a **Machine Learning model** trained on the TMDB dataset to  
predict the **rating** (`vote_average`) of a movie based on:

- Genre  
- Main actor  
- Director  
- Popularity, vote count, release year
"""
)

# Load data and model
df, X, y, numeric_features, categorical_features = load_and_engineer()
model, metrics = train_model(X, y, numeric_features, categorical_features)

st.success("Model ready!")


# Movie selection
movie_titles = df["title"].dropna().unique()
movie_titles = np.sort(movie_titles)

selected_title = st.selectbox(
    "Pick a movie from the TMDB dataset:",
    movie_titles,
)

# When user chooses a movie
if selected_title:
    row = df[df["title"] == selected_title].iloc[0]

    # build a single-row dataframe with the same feature columns as training
    feature_row = pd.DataFrame(
        [{
            "popularity": row["popularity"],
            "runtime": row["runtime"],
            "budget_log": row["budget_log"],
            "vote_count": row["vote_count"],
            "release_year": row["release_year"],
            "main_genre": row["main_genre"],
            "main_actor": row["main_actor"],
            "director_name": row["director_name"],            
        }]
    )

    pred_rating = float(model.predict(feature_row)[0])
    true_rating = float(row["vote_average"])

    # layout: poster on left, info on right
    col1, col2 = st.columns([1, 2])

    with col1:
        poster_url = fetch_poster(row["id"])
        if poster_url:
            st.image(poster_url, caption=selected_title, width="stretch")
        else:
            st.write("No poster available for this movie.")

    with col2:
        st.subheader(selected_title)
        st.markdown(f"**Predicted rating:** {pred_rating:.2f}")
        st.markdown(f"**TMDB rating (vote_average):** {true_rating:.2f}")

        st.markdown(f"**Main genre:** {row['main_genre']}")
        st.markdown(f"**Main actor:** {row['main_actor']}")
        st.markdown(f"**Director:** {row['director_name']}")
        st.markdown(f"**Release year:** {int(row['release_year']) if not pd.isna(row['release_year']) else 'Unknown'}")

        st.markdown(
            f"**Popularity:** {row['popularity']:.2f} &nbsp;&nbsp;"
            f"**Vote count:** {int(row['vote_count'])}"
        )    

   
