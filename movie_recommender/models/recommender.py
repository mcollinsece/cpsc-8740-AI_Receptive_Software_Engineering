import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

class MovieRecommender:
    def __init__(self, data_processor):
        """
        Initialize the recommender with a data processor.
        
        Args:
            data_processor (DataProcessor): Instance of DataProcessor class
        """
        self.data_processor = data_processor
        self.user_similarity = None
        self.item_similarity = None
        self.user_movie_matrix = None
        
    def create_user_movie_matrix(self):
        """Create and normalize the user-movie rating matrix."""
        # Create the user-movie matrix
        self.user_movie_matrix = self.data_processor.ratings.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        # Normalize the matrix by subtracting user means
        user_means = self.user_movie_matrix.mean(axis=1)
        self.user_movie_matrix = self.user_movie_matrix.sub(user_means, axis=0)
        
    def compute_similarities(self):
        """Compute user and item similarity matrices."""
        if self.user_movie_matrix is None:
            self.create_user_movie_matrix()
            
        # Compute user similarity
        self.user_similarity = cosine_similarity(self.user_movie_matrix)
        
        # Compute item similarity
        self.item_similarity = cosine_similarity(self.user_movie_matrix.T)
        
    def get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations using collaborative filtering."""
        if self.user_similarity is None:
            self.compute_similarities()
            
        # Get user index
        user_idx = self.user_movie_matrix.index.get_loc(user_id)
        
        # Get movies not rated by user
        user_ratings = self.user_movie_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            movie_idx = self.user_movie_matrix.columns.get_loc(movie_id)
            
            # Get similar users who rated this movie
            similar_users = self.user_similarity[user_idx]
            movie_ratings = self.user_movie_matrix.iloc[:, movie_idx]
            rated_by_similar = movie_ratings != 0
            
            if rated_by_similar.any():
                # Weighted average of ratings from similar users
                pred = np.average(
                    movie_ratings[rated_by_similar],
                    weights=similar_users[rated_by_similar]
                )
                predictions.append((movie_id, pred))
            
        # Sort by predicted rating and return top n
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
        
    def get_content_based_recommendations(self, movie_id, n_recommendations=10):
        """Get recommendations using content-based filtering."""
        similar_movies = self.data_processor.get_movie_similarity(movie_id)
        return similar_movies[:n_recommendations]
        
    def get_hybrid_recommendations(self, user_id, movie_id, n_recommendations=10):
        """Get recommendations using a hybrid approach."""
        # Get content-based recommendations
        content_recs = self.get_content_based_recommendations(movie_id)
        
        # Get collaborative filtering predictions for these movies
        hybrid_recs = []
        for movie_idx, similarity in content_recs:
            movie_id = self.data_processor.movies.iloc[movie_idx]['movieId']
            # Get collaborative prediction
            collab_pred = self.get_collaborative_recommendations(user_id, n_recommendations=1)
            if collab_pred:
                pred_score = collab_pred[0][1] * similarity
                hybrid_recs.append((movie_id, pred_score))
            
        # Sort by hybrid score and return top n
        hybrid_recs.sort(key=lambda x: x[1], reverse=True)
        return hybrid_recs[:n_recommendations]
        
    def evaluate_model(self):
        """Evaluate the model using RMSE and MAE."""
        if self.user_movie_matrix is None:
            self.create_user_movie_matrix()
            
        # Split data into train and test sets
        train_data, test_data = train_test_split(
            self.data_processor.ratings,
            test_size=0.2,
            random_state=42
        )
        
        # Create training matrix
        train_matrix = train_data.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        # Compute user means for normalization
        user_means = train_matrix.mean(axis=1)
        train_matrix = train_matrix.sub(user_means, axis=0)
        
        # Compute user similarity
        user_similarity = cosine_similarity(train_matrix)
        
        # Make predictions for test set
        predictions = []
        actual_ratings = []
        
        for _, row in test_data.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            if user_id in train_matrix.index and movie_id in train_matrix.columns:
                user_idx = train_matrix.index.get_loc(user_id)
                movie_idx = train_matrix.columns.get_loc(movie_id)
                
                # Get similar users who rated this movie
                similar_users = user_similarity[user_idx]
                movie_ratings = train_matrix.iloc[:, movie_idx]
                rated_by_similar = movie_ratings != 0
                
                if rated_by_similar.any():
                    pred = np.average(
                        movie_ratings[rated_by_similar],
                        weights=similar_users[rated_by_similar]
                    )
                    predictions.append(pred)
                    actual_ratings.append(actual_rating)
        
        # Calculate RMSE and MAE
        predictions = np.array(predictions)
        actual_ratings = np.array(actual_ratings)
        
        rmse = np.sqrt(np.mean((predictions - actual_ratings) ** 2))
        mae = np.mean(np.abs(predictions - actual_ratings))
        
        return {'RMSE': rmse, 'MAE': mae} 