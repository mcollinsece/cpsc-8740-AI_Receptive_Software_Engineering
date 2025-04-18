from flask import Flask, render_template, request, jsonify
import os
import sys

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
            'recommendations': movie_details
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