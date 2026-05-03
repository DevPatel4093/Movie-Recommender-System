# 🎬 Movie Recommendation System

A content-based movie recommender system built using Python, Pandas, and scikit-learn. The system suggests movies based on similarity in genres, overview, keywords, and cast information.

## 🚀 Features

- Recommends top 5 similar movies based on a selected title.
- Uses **TF-IDF** and **cosine similarity** for content-based filtering.
- Clean and interactive frontend using **Streamlit**.
- Movie posters fetched dynamically using the **TMDB API**.

## 📊 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Libraries**: Pandas, NumPy, scikit-learn, requests, pickle
- **Data**: TMDB 5000 Movie Dataset (`tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`)

## 📁 Project Structure
```
Movie-Rating-Prediction-Model/
├── tmdb_5000_movies.csv   
├── tmdb_5000_credits.csv                
├── prediction_model.py                          
├── movie_ratings_with_predictions.csv                             
├── rate_app.py // Streamlit UI for Prediction with Posters                     
├── rate.py // Streamlit UI for Prediction without Posters                        
├── docker-compose-dev.yml           
├── How_To_Run.txt                                 
└── README.md 
```

## 🧠 How It Works

1. **Preprocessing**: Merge movie and credit data, extract keywords, genres, cast, and crew.
2. **Feature Engineering**: Combine relevant features into a single text field.
3. **Vectorization**: Use TF-IDF or CountVectorizer to convert text into numerical format.
4. **Similarity Scoring**: Apply cosine similarity to identify most similar movies.
5. **Display**: Render recommendations with posters and titles using Streamlit UI.

<<<<<<< Updated upstream
## 👨‍💻 Author
**Dev Patel**
=======
## 👤 Author
**Dev Patel**
>>>>>>> Stashed changes
