import streamlit as st
import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommendation System")

# --- Functions ---
@st.cache_data
def load_and_preprocess():
    # Load your local files
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = pd.read_csv('tmdb_5000_movies.csv')
    
    # Merge and Clean
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])
    
    def fetch_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director': return [i['name']]
        return []

    movies['crew'] = movies['crew'].apply(fetch_director)
    
    # Text Processing for Tags
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
    
    return new_df

@st.cache_resource
def calculate_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity

# --- Main Execution ---
try:
    df = load_and_preprocess()
    similarity = calculate_similarity(df)

    # UI Components
    selected_movie = st.selectbox(
        "Type or select a movie to get recommendations:",
        df['title'].values
    )

    if st.button('Show Recommendation'):
        idx = df[df['title'] == selected_movie].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])[1:11]
        
        st.subheader(f"Movies similar to {selected_movie}:")
        
        # Display recommendations in a grid
        cols = st.columns(5)
        for i, movie_info in enumerate(distances[:5]):
            with cols[i]:
                st.text(df.iloc[movie_info[0]].title)
                
        cols2 = st.columns(5)
        for i, movie_info in enumerate(distances[5:10]):
            with cols2[i]:
                st.text(df.iloc[movie_info[0]].title)

except FileNotFoundError:
    st.error("Dataset files not found. Please ensure tmdb_5000_movies.csv and tmdb_5000_credits.csv are in the folder.")
