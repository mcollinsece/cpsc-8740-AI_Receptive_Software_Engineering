from flask import Flask, request, jsonify
import torch
import json
import logging
from models.recommender import Recommender
import os

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
        checkpoint = torch.load('recommender_model.pth', map_location='cpu')  # Load to CPU first
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

        # Check if user exists in training data
        if user_id >= model.user_embedding.num_embeddings:
            return jsonify({
                'error': 'New user detected. Please rate some movies first.',
                'suggestion': 'Rate these popular movies to get started: ...'
            }), 400

        # Convert movie IDs to strings for dictionary lookup
        movie_ids = [str(mid) for mid in movie_ids]
        
        # Convert movie IDs to indices
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
            rating = float(predicted_ratings[idx].item())  # Convert to float for JSON serialization
            title = mappings['moviesid_to_title'][str(movie_id)]  # Convert ID to string
            recommendations.append({
                'movie_id': movie_id,
                'title': title,
                'predicted_rating': rating
            })

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        logger.error(f"Error in recommend endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)  # Enable debug mode for development 