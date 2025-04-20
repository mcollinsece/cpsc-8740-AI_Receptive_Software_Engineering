import torch
import json
import logging
from models.recommender import Recommender
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RecommenderService:
    def __init__(self):
        self.model = None
        self.mappings = None
        self.device = None
        self._load_model()

    def _load_model(self):
        """Load the model and mappings from disk"""
        try:
            # Check if model file exists
            if not os.path.exists('recommender_model.pth'):
                raise FileNotFoundError("recommender_model.pth not found. Please run train.py first.")
            
            if not os.path.exists('model_mappings.json'):
                raise FileNotFoundError("model_mappings.json not found. Please run train.py first.")

            # Load model
            checkpoint = torch.load('recommender_model.pth', map_location='cpu')  # Load to CPU first
            self.model = Recommender(checkpoint['num_users'], checkpoint['num_items'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load mappings
            with open('model_mappings.json', 'r') as f:
                self.mappings = json.load(f)
            
            # Set device
            self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            self.model = self.model.to(self.device)

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_recommendations(self, user_id, movie_ids, top_n=10):
        """
        Get movie recommendations for a user
        
        Args:
            user_id (int): The ID of the user
            movie_ids (list): List of movie IDs to get predictions for
            top_n (int): Number of top recommendations to return
            
        Returns:
            list: List of dictionaries containing movie recommendations
        """
        try:
            # Check if user exists in training data
            if user_id >= self.model.user_embedding.num_embeddings:
                raise ValueError('New user detected. Please rate some movies first.')

            # Convert movie IDs to strings for dictionary lookup
            movie_ids = [str(mid) for mid in movie_ids]
            
            # Convert movie IDs to indices
            valid_movie_indices = []
            for mid in movie_ids:
                if mid in self.mappings['movie_id_to_idx']:
                    valid_movie_indices.append(self.mappings['movie_id_to_idx'][mid])
            
            if not valid_movie_indices:
                raise ValueError('No valid movie IDs provided')

            # Prepare tensors
            user = torch.tensor([user_id] * len(valid_movie_indices), device=self.device)
            movies = torch.tensor(valid_movie_indices, device=self.device)

            # Get predictions
            with torch.no_grad():
                predicted_ratings = self.model(user, movies).squeeze()

            # Get top N recommendations
            top_indices = predicted_ratings.argsort(descending=True)[:top_n]
            
            recommendations = []
            for idx in top_indices:
                movie_idx = valid_movie_indices[idx]
                movie_id = self.mappings['idx_to_movie_id'][str(movie_idx)]
                rating = float(predicted_ratings[idx].item())  # Convert to float for JSON serialization
                title = self.mappings['moviesid_to_title'][str(movie_id)]  # Convert ID to string
                recommendations.append({
                    'movie_id': movie_id,
                    'title': title,
                    'predicted_rating': rating
                })

            return recommendations

        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}", exc_info=True)
            raise

# Example usage
if __name__ == '__main__':
    # Initialize the service
    recommender = RecommenderService()
    
    # Example: Get recommendations for user 5
    try:
        recommendations = recommender.get_recommendations(
            user_id=5,
            movie_ids=[1, 2, 3, 4, 5],
            top_n=3
        )
        print("Recommendations:", json.dumps(recommendations, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}") 