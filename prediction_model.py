import ast
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------
# 1. Load & merge datasets
# ---------------------------
print("Loading data...")
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

credits.rename(columns={"movie_id": "id"}, inplace=True)
df = movies.merge(credits, on="id", how="left")
print("Merged shape:", df.shape)


# ---------------------------
# 2. Helper functions
# ---------------------------
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


# ---------------------------
# 3. Feature engineering
# ---------------------------

print("Engineering features...")

# basic numeric features
df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")
df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")

df["budget_log"] = np.log1p(df["budget"])  # log(1+budget), handles 0 and NaN

# date -> year
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = df["release_date"].dt.year

# categorical features from lists
df["main_genre"] = df["genres"].apply(parse_genre_first)
df["main_actor"] = df["cast"].apply(parse_cast_first)
df["director_name"] = df["crew"].apply(get_director)

# target
target_col = "vote_average"
df = df[~df[target_col].isna()].copy()  # drop rows with missing rating

# ---------------------------
# 4. Select final features
# ---------------------------

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
y = df[target_col]

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)


# ---------------------------
# 5. Build preprocessing + model pipeline
# ---------------------------

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
    n_jobs=-1
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)


# ---------------------------
# 6. Train / test split & fit
# ---------------------------

print("Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training model (this may take a bit)...")
pipe.fit(X_train, y_train)
print("Training complete.")


# ---------------------------
# 7. Evaluation
# ---------------------------

print("Evaluating...")
y_pred = pipe.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))   
r2 = r2_score(y_test, y_pred)

print("\nModel performance on test set:")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ---------------------------
# 8. Example prediction
# ---------------------------

example = X.iloc[0:1].copy()
example_pred = pipe.predict(example)[0]

print("\nExample movie:")
print(example[numeric_features + categorical_features])
print(f"\nPredicted rating: {example_pred:.3f}")
print(f"True rating     : {y.iloc[0]:.3f}")


# ---------------------------
# 9. Predict rating for ALL movies
# ---------------------------

print("\nPredicting ratings for ALL movies...")

# retrain model on full dataset (recommended for final predictions)
pipe.fit(X, y)

# get prediction for every movie
all_predictions = pipe.predict(X)

# attach predictions to dataframe
df["predicted_rating"] = all_predictions

# show first few rows
print("\nSample of predicted ratings:")
print(df[["title_x", "vote_average", "predicted_rating"]])
#print(df[["title_x", "vote_average", "predicted_rating"]].head(10))

# save output to CSV
df[["id", "title_x", "vote_average", "predicted_rating"]].to_csv(
    "movie_ratings_with_predictions.csv",
    index=False
)

print("\nSaved as: movie_ratings_with_predictions.csv")