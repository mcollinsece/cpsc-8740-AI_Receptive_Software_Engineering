import json
import torch
import os
from models.recommender import Recommender

# Initialize the model and mappings outside the handler to reuse between invocations
model = None
mappings = None
device = None

def load_model():
    global model, mappings, device
    if model is None:  # Only load if not already loaded
        try:
            # Load model
            checkpoint = torch.load('recommender_model.pth', map_location='cpu')
            model = Recommender(checkpoint['num_users'], checkpoint['num_items'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Load mappings
            with open('model_mappings.json', 'r') as f:
                mappings = json.load(f)
            
            # Set device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

def get_recommendations(user_id, movie_ids, top_n=10):
    try:
        # Check if user exists in training data
        if user_id >= model.user_embedding.num_embeddings:
            raise ValueError('New user detected. Please rate some movies first.')

        # Convert movie IDs to strings for dictionary lookup
        movie_ids = [str(mid) for mid in movie_ids]
        
        # Convert movie IDs to indices
        valid_movie_indices = []
        for mid in movie_ids:
            if mid in mappings['movie_id_to_idx']:
                valid_movie_indices.append(mappings['movie_id_to_idx'][mid])
        
        if not valid_movie_indices:
            raise ValueError('No valid movie IDs provided')

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

        return recommendations

    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        raise

def lambda_handler(event, context):
    # Load model on first invocation
    load_model()
    
    try:
        # Parse input
        body = json.loads(event['body'])
        user_id = body['user_id']
        movie_ids = body['movie_ids']
        top_n = body.get('top_n', 10)
        
        # Get recommendations
        recommendations = get_recommendations(user_id, movie_ids, top_n)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'recommendations': recommendations
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        } 