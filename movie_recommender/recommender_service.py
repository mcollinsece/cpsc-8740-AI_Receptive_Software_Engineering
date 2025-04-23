from flask import Flask, request, jsonify
import torch
import json
import logging
import os
from models.recommender import Recommender

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class RecommenderService:
    """
    A service class that handles movie recommendations using a trained model.
    This class manages the model loading, prediction, and recommendation generation.
    """
    def __init__(self):
        """Initialize the service by setting up model and device."""
        self.model = None  # Will hold the trained recommender model
        self.mappings = None  # Will hold the ID mappings
        self.device = None  # Will hold the device (CPU/GPU) for model execution
        self._load_model()  # Load the model and mappings

    def _load_model(self):
        """
        Load the trained model and mappings from disk.
        This method:
        1. Checks if required files exist
        2. Loads the model checkpoint
        3. Initializes the Recommender model
        4. Loads the ID mappings
        5. Sets up the appropriate device (CPU/GPU)
        """
        try:
            # Check if model file exists
            if not os.path.exists('recommender_model.pth'):
                raise FileNotFoundError("recommender_model.pth not found. Please run train.py first.")
            
            if not os.path.exists('model_mappings.json'):
                raise FileNotFoundError("model_mappings.json not found. Please run train.py first.")

            # Load model checkpoint
            checkpoint = torch.load('recommender_model.pth', map_location='cpu')  # Load to CPU first
            self.model = Recommender(checkpoint['num_users'], checkpoint['num_items'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Set model to evaluation mode
            
            # Load ID mappings
            with open('model_mappings.json', 'r') as f:
                self.mappings = json.load(f)
            
            # Set device (prefer GPU if available)
            self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            self.model = self.model.to(self.device)

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_recommendations(self, user_id, movie_ids=None, top_n=10):
        """
        Get movie recommendations for a user.
        
        Args:
            user_id (int): The ID of the user to get recommendations for
            movie_ids (list, optional): List of movie IDs to consider. If None, considers all movies.
            top_n (int): Number of top recommendations to return
            
        Returns:
            list: List of dictionaries containing movie recommendations with:
                - movie_id: The ID of the movie
                - title: The title of the movie
                - predicted_rating: The predicted rating for this user
                
        Raises:
            ValueError: If user is not found or no valid movies are provided
        """
        try:
            # Check if user exists in training data
            if user_id >= self.model.user_embedding.num_embeddings:
                raise ValueError('New user detected. Please rate some movies first.')

            if movie_ids is None:
                # Case 1: Get recommendations from all movies
                all_movie_indices = list(range(self.model.item_embedding.num_embeddings))
                user = torch.tensor([user_id] * len(all_movie_indices), device=self.device)
                movies = torch.tensor(all_movie_indices, device=self.device)
                
                # Get predictions for all movies
                with torch.no_grad():
                    predicted_ratings = self.model(user, movies).squeeze()
                
                # Get top N recommendations
                top_indices = predicted_ratings.argsort(descending=True)[:top_n]
                
                # Format recommendations
                recommendations = []
                for idx in top_indices:
                    movie_idx = all_movie_indices[idx]
                    movie_id = self.mappings['idx_to_movie_id'][str(movie_idx)]
                    rating = float(predicted_ratings[idx].item())
                    title = self.mappings['moviesid_to_title'][str(movie_id)]
                    recommendations.append({
                        'movie_id': movie_id,
                        'title': title,
                        'predicted_rating': rating
                    })
            else:
                # Case 2: Get recommendations from specific movies
                movie_ids = [str(mid) for mid in movie_ids]
                
                # Convert movie IDs to model indices
                valid_movie_indices = []
                for mid in movie_ids:
                    if mid in self.mappings['movie_id_to_idx']:
                        valid_movie_indices.append(self.mappings['movie_id_to_idx'][mid])
                
                if not valid_movie_indices:
                    raise ValueError('No valid movie IDs provided')

                # Prepare tensors for prediction
                user = torch.tensor([user_id] * len(valid_movie_indices), device=self.device)
                movies = torch.tensor(valid_movie_indices, device=self.device)

                # Get predictions
                with torch.no_grad():
                    predicted_ratings = self.model(user, movies).squeeze()

                # Get top N recommendations
                top_indices = predicted_ratings.argsort(descending=True)[:top_n]
                
                # Format recommendations
                recommendations = []
                for idx in top_indices:
                    movie_idx = valid_movie_indices[idx]
                    movie_id = self.mappings['idx_to_movie_id'][str(movie_idx)]
                    rating = float(predicted_ratings[idx].item())
                    title = self.mappings['moviesid_to_title'][str(movie_id)]
                    recommendations.append({
                        'movie_id': movie_id,
                        'title': title,
                        'predicted_rating': rating
                    })

            return recommendations

        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}", exc_info=True)
            raise

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Flask endpoint for getting movie recommendations.
    
    Expected JSON input:
    {
        "user_id": int,  # Required
        "movie_ids": [int, ...],  # Optional
        "top_n": int  # Optional, defaults to 10
    }
    
    Returns:
        JSON response with recommendations or error message
    """
    try:
        # Parse request data
        data = request.get_json()
        user_id = data.get('user_id')
        movie_ids = data.get('movie_ids')  # Optional
        top_n = data.get('top_n', 10)  # Default to 10 if not specified

        # Validate required parameters
        if not user_id:
            return jsonify({'error': 'Missing required parameter: user_id'}), 400

        # Initialize service if needed
        if not hasattr(app, 'recommender'):
            app.recommender = RecommenderService()

        # Get recommendations
        recommendations = app.recommender.get_recommendations(user_id, movie_ids, top_n)

        # Return response
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations
        })

    except Exception as e:
        # Handle errors
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000) 