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


# ===========================
# 0. TMDB API CONFIG
# ===========================
# 👉 PUT YOUR REAL TMDB API KEY HERE
# TMDB_API_KEY = "YOUR_REAL_API_KEY_HERE"    # <-- replace this with your key

TMDB_API_KEY = "b84298fc0f62ad211030847c042d8717"    # <-- replace this with your key

TMDB_BASE_URL = "https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US"
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w500"


def fetch_poster(movie_id: int):
    """
    Given TMDB movie id, return full poster URL.
    If request fails or poster not available, return None.
    """
    if not TMDB_API_KEY or TMDB_API_KEY == "YOUR_REAL_API_KEY_HERE":
        # no key set: skip calling TMDB
        return None

    try:
        url = TMDB_BASE_URL.format(movie_id, TMDB_API_KEY)
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return None
        return TMDB_POSTER_BASE + poster_path
    except Exception:
        # network / SSL / bad id etc.
        return None


# ===========================
# 1. HELPER FUNCTIONS (same logic as prediction_model.py)
# ===========================

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


# GLOBAL feature lists (so we can reuse in UI)
NUMERIC_FEATURES = [
    "popularity",
    "runtime",
    "budget_log",
    "vote_count",
    "release_year",
]

CATEGORICAL_FEATURES = [
    "main_genre",
    "main_actor",
    "director_name",
]

TARGET_COL = "vote_average"


# ===========================
# 2. LOAD DATA + TRAIN MODEL
# ===========================

@st.cache_resource
def load_data_and_model():
    """
    Loads TMDB data, engineers features, trains RandomForest model,
    returns (df, X, y, pipe, metrics_dict)
    """
    st.write("🔄 Loading data and training model (first run only)...")

    # ---- Load & merge ----
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits, on="id", how="left")

    # rename title_x -> title (fix your KeyError 'title')
    if "title_x" in df.columns:
        df = df.rename(columns={"title_x": "title"})
    # we don't need title_y
    if "title_y" in df.columns:
        df = df.drop(columns=["title_y"])

    # ---- Feature engineering (same as prediction_model.py) ----
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")

    df["budget_log"] = np.log1p(df["budget"])  # log(1+budget)

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    df["main_genre"] = df["genres"].apply(parse_genre_first)
    df["main_actor"] = df["cast"].apply(parse_cast_first)
    df["director_name"] = df["crew"].apply(get_director)

    # drop rows with missing target
    df = df[~df[TARGET_COL].isna()].copy()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    # ---- Build preprocessing + model ----
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
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # ---- Train / test split & fit ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # ---- Also compute predicted rating for ALL movies ----
    df["predicted_rating"] = pipe.predict(X)

    return df, X, y, pipe, metrics


# ===========================
# 3. STREAMLIT UI
# ===========================

st.set_page_config(page_title="Movie Rating Prediction", layout="wide")

st.title("🎬 Movie Rating Prediction")
st.markdown("Predict movie ratings using **genre, director, actors**, and show **TMDB posters**.")

# load model & data
df, X, y, pipe, metrics = load_data_and_model()

# Sidebar metrics
with st.sidebar:
    st.header("📊 Model Performance")
    st.write(f"**MAE**  : `{metrics['MAE']:.4f}`")
    st.write(f"**RMSE** : `{metrics['RMSE']:.4f}`")
    st.write(f"**R²**   : `{metrics['R2']:.4f}`")
    st.write(f"Train size: `{metrics['n_train']}`")
    st.write(f"Test size : `{metrics['n_test']}`")

# Movie selector
st.markdown("### 🎥 Choose a movie")
movie_titles = df["title"].sort_values().unique()
selected_title = st.selectbox("Select a movie title", movie_titles)

if st.button("🔮 Predict Rating"):
    # find selected movie row
    row = df[df["title"] == selected_title].iloc[0]

    # build single-row dataframe for prediction
    single_X = pd.DataFrame(
        {col: [row[col]] for col in (NUMERIC_FEATURES + CATEGORICAL_FEATURES)}
    )

    predicted_rating = float(pipe.predict(single_X)[0])
    true_rating = float(row[TARGET_COL])
    movie_id = int(row["id"])

    # Layout: left = info+poster, right = metrics
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader(selected_title)
        st.markdown(f"**Main genre:** `{row['main_genre']}`")
        st.markdown(f"**Main actor:** `{row['main_actor']}`")
        st.markdown(f"**Director:** `{row['director_name']}`")
        st.markdown(f"**Year:** `{int(row['release_year']) if not np.isnan(row['release_year']) else 'Unknown'}`")

        poster_url = fetch_poster(movie_id)
        if poster_url:
            st.image(poster_url, caption=selected_title, width=300)
        else:
            st.info("Poster not available (no API key or no poster for this movie).")

    with col_right:
        st.subheader("⭐ Rating comparison")
        st.markdown("---")
        st.markdown(f"🎯 **True rating (dataset):** `{true_rating:.2f}`")
        st.markdown(f"🤖 **Predicted rating (model):** `{predicted_rating:.2f}`")
        diff = predicted_rating - true_rating
        st.write(f"Difference (pred - true): `{diff:.3f}`")

    # Show similar movies by predicted rating (your requested snippet)
    st.markdown("### 🔎 Movies with similar predicted rating")
    # we already computed df['predicted_rating'] in load_data_and_model
    df["pred_diff"] = (df["predicted_rating"] - predicted_rating).abs()
    similar_df = df.sort_values("pred_diff").head(5)[
        ["title", "vote_average", "predicted_rating"]
    ]
    st.dataframe(similar_df.reset_index(drop=True))
