from flask import Flask, request, jsonify
import torch
import json
import logging
from models.recommender import Recommender
import os
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model and mappings
def load_model():
    try:
        # Check if model file exists
        if not os.path.exists('recommender_model.pth'):
            raise FileNotFoundError("recommender_model.pth not found. Please run train.py first.")
        
        if not os.path.exists('model_mappings.json'):
            raise FileNotFoundError("model_mappings.json not found. Please run train.py first.")

        # Load model
        checkpoint = torch.load('recommender_model.pth', map_location='cpu')
        model = Recommender(checkpoint['num_users'], checkpoint['num_items'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load mappings
        with open('model_mappings.json', 'r') as f:
            mappings = json.load(f)
        
        return model, mappings

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

try:
    model, mappings = load_model()
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    model = model.to(device)
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise

# Store new user ratings
new_user_ratings = {}

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        logger.debug(f"Received request data: {data}")

        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        user_id = data.get('user_id')
        movie_ids = data.get('movie_ids')
        top_n = data.get('top_n', 10)

        if user_id is None or movie_ids is None:
            return jsonify({'error': 'Missing required parameters: user_id or movie_ids'}), 400

        # Check if user is new
        if user_id >= model.user_embedding.num_embeddings:
            # For new users, use content-based filtering
            if user_id not in new_user_ratings:
                new_user_ratings[user_id] = []
            
            # Get content-based recommendations
            recommendations = get_content_based_recommendations(movie_ids, top_n)
            
            return jsonify({
                'recommendations': recommendations,
                'is_new_user': True,
                'message': 'New user detected. Showing content-based recommendations. Rate some movies to get personalized recommendations.'
            })

        # For existing users, use the model
        movie_ids = [str(mid) for mid in movie_ids]
        valid_movie_indices = []
        for mid in movie_ids:
            if mid in mappings['movie_id_to_idx']:
                valid_movie_indices.append(mappings['movie_id_to_idx'][mid])
        
        if not valid_movie_indices:
            return jsonify({'error': 'No valid movie IDs provided'}), 400

        # Prepare tensors
        user = torch.tensor([user_id] * len(valid_movie_indices), device=device)
        movies = torch.tensor(valid_movie_indices, device=device)

        # Get predictions
        with torch.no_grad():
            predicted_ratings = model(user, movies).squeeze()

        # Get top N recommendations
        top_indices = predicted_ratings.argsort(descending=True)[:top_n]
        
        recommendations = []
        for idx in top_indices:
            movie_idx = valid_movie_indices[idx]
            movie_id = mappings['idx_to_movie_id'][str(movie_idx)]
            rating = float(predicted_ratings[idx].item())
            title = mappings['moviesid_to_title'][str(movie_id)]
            recommendations.append({
                'movie_id': movie_id,
                'title': title,
                'predicted_rating': rating
            })

        return jsonify({
            'recommendations': recommendations,
            'is_new_user': False
        })

    except Exception as e:
        logger.error(f"Error in recommend endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/rate', methods=['POST'])
def rate_movie():
    try:
        data = request.json
        user_id = data.get('user_id')
        movie_id = data.get('movie_id')
        rating = data.get('rating')

        if not all([user_id, movie_id, rating]):
            return jsonify({'error': 'Missing required parameters'}), 400

        # Check if user is new
        if user_id >= model.user_embedding.num_embeddings:
            if user_id not in new_user_ratings:
                new_user_ratings[user_id] = []
            
            # Add the rating
            new_user_ratings[user_id].append({
                'movie_id': movie_id,
                'rating': rating
            })

            # If user has rated enough movies, update the model
            if len(new_user_ratings[user_id]) >= 10:
                update_model_with_new_user(user_id)
                return jsonify({
                    'message': 'User profile updated. Now using collaborative filtering.',
                    'ratings_count': len(new_user_ratings[user_id])
                })
            
            return jsonify({
                'message': f'Rating recorded. {10 - len(new_user_ratings[user_id])} more ratings needed for personalized recommendations.',
                'ratings_count': len(new_user_ratings[user_id])
            })
        
        return jsonify({'error': 'User already exists in the system'}), 400

    except Exception as e:
        logger.error(f"Error in rate endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def get_content_based_recommendations(movie_ids, top_n):
    """Get content-based recommendations for new users."""
    # This is a placeholder - implement your content-based filtering logic here
    # For example, you could use genre similarity, movie metadata, etc.
    recommendations = []
    for movie_id in movie_ids[:top_n]:
        if str(movie_id) in mappings['moviesid_to_title']:
            recommendations.append({
                'movie_id': movie_id,
                'title': mappings['moviesid_to_title'][str(movie_id)],
                'predicted_rating': 4.0  # Placeholder rating
            })
    return recommendations

def update_model_with_new_user(user_id):
    """Update the model with a new user's ratings."""
    # This is a placeholder - implement your model update logic here
    # For example, you could retrain the model or update the embeddings
    pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050) 