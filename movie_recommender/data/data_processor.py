import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DataProcessor:
    def __init__(self, movies_path, ratings_path):
        """
        Initialize the data processor with paths to the MovieLens dataset files.
        
        Args:
            movies_path (str): Path to the movies.csv file
            ratings_path (str): Path to the ratings.csv file
        """
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        self.movie_features = None
        self.user_movie_matrix = None
        self.movie_similarity = None
        
        # Initialize similarity matrix
        self.preprocess_movies()
        
    def preprocess_movies(self):
        """Preprocess the movies data and create TF-IDF features."""
        # Convert genres into a format suitable for TF-IDF
        self.movies['genres'] = self.movies['genres'].str.replace('|', ' ')
        
        # Create TF-IDF features for genres
        tfidf = TfidfVectorizer(stop_words='english')
        self.movie_features = tfidf.fit_transform(self.movies['genres'])
        
        # Compute movie similarity matrix
        self.movie_similarity = cosine_similarity(self.movie_features)
        
    def create_user_movie_matrix(self):
        """Create the user-movie rating matrix."""
        self.user_movie_matrix = self.ratings.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
    def get_movie_similarity(self, movie_id):
        """Get similar movies based on content."""
        if self.movie_similarity is None:
            self.preprocess_movies()
            
        movie_idx = self.movies[self.movies['movieId'] == movie_id].index[0]
        similar_movies = list(enumerate(self.movie_similarity[movie_idx]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        
        return similar_movies[1:11]  # Return top 10 similar movies
        
    def get_user_ratings(self, user_id):
        """Get all ratings for a specific user."""
        return self.ratings[self.ratings['userId'] == user_id]
        
    def get_movie_info(self, movie_id):
        """Get movie information."""
        return self.movies[self.movies['movieId'] == movie_id] 