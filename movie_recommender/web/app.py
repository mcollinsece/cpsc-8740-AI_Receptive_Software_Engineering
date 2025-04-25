from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_processor import DataProcessor
from models.recommender import MovieRecommender

app = Flask(__name__)

# Initialize data processor and recommender
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
movies_path = os.path.join(data_dir, 'movies.csv')
ratings_path = os.path.join(data_dir, 'ratings.csv')

data_processor = DataProcessor(movies_path, ratings_path)
recommender = MovieRecommender(data_processor)

# Store new user ratings
new_user_ratings = {}

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Get movie recommendations based on user input."""
    data = request.json
    user_id = int(data.get('user_id'))
    movie_id = int(data.get('movie_id'))
    method = data.get('method', 'hybrid')
    
    try:
        # Check if user is new (not in training data)
        is_new_user = user_id not in data_processor.ratings['userId'].unique()
        
        if is_new_user:
            # For new users, first get content-based recommendations
            initial_recommendations = recommender.get_content_based_recommendations(movie_id)
            
            # If this is the first time for this user, initialize their ratings
            if user_id not in new_user_ratings:
                new_user_ratings[user_id] = []
            
            # Store the initial recommendation
            new_user_ratings[user_id].append({
                'movie_id': movie_id,
                'rating': 5.0  # Assuming they liked the movie they searched for
            })
            
            # Get movie details for recommendations
            movie_details = []
            for movie_idx, similarity in initial_recommendations:
                movie_id = data_processor.movies.iloc[movie_idx]['movieId']
                movie_info = data_processor.get_movie_info(movie_id)
                if not movie_info.empty:
                    movie_details.append({
                        'title': movie_info['title'].iloc[0],
                        'genres': movie_info['genres'].iloc[0],
                        'score': round(similarity, 2)
                    })
            
            return jsonify({
                'success': True,
                'recommendations': movie_details,
                'is_new_user': True,
                'message': 'New user detected. Showing content-based recommendations. Rate some movies to get personalized recommendations.'
            })
        else:
            # For existing users, use the standard recommendation methods
            if method == 'collaborative':
                recommendations = recommender.get_collaborative_recommendations(user_id)
            elif method == 'content':
                recommendations = recommender.get_content_based_recommendations(movie_id)
            else:
                recommendations = recommender.get_hybrid_recommendations(user_id, movie_id)
            
            # Get movie details for recommendations
            movie_details = []
            for movie_id, score in recommendations:
                movie_info = data_processor.get_movie_info(movie_id)
                if not movie_info.empty:
                    movie_details.append({
                        'title': movie_info['title'].iloc[0],
                        'genres': movie_info['genres'].iloc[0],
                        'score': round(score, 2)
                    })
            
            return jsonify({
                'success': True,
                'recommendations': movie_details,
                'is_new_user': False
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/rate', methods=['POST'])
def rate_movie():
    """Handle movie ratings from new users."""
    data = request.json
    user_id = int(data.get('user_id'))
    movie_id = int(data.get('movie_id'))
    rating = float(data.get('rating'))
    
    try:
        if user_id in new_user_ratings:
            # Add the rating to the user's history
            new_user_ratings[user_id].append({
                'movie_id': movie_id,
                'rating': rating
            })
            
            # If user has rated enough movies, update the training data
            if len(new_user_ratings[user_id]) >= 10:
                # Create new ratings DataFrame
                new_ratings = pd.DataFrame(new_user_ratings[user_id])
                new_ratings['userId'] = user_id
                
                # Append to existing ratings
                data_processor.ratings = pd.concat([
                    data_processor.ratings,
                    new_ratings[['userId', 'movie_id', 'rating']].rename(columns={'movie_id': 'movieId'})
                ])
                
                # Recompute similarities
                recommender.compute_similarities()
                
                return jsonify({
                    'success': True,
                    'message': 'User profile updated. Now using collaborative filtering.'
                })
            
            return jsonify({
                'success': True,
                'message': f'Rating recorded. {10 - len(new_user_ratings[user_id])} more ratings needed for personalized recommendations.'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'User not found'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/evaluate', methods=['GET'])
def evaluate():
    """Evaluate the model and return metrics."""
    try:
        metrics = recommender.evaluate_model()
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 