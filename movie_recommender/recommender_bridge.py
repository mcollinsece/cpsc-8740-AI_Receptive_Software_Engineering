import sys
import json
from recommender_service import RecommenderService

def main():
    # Initialize the service
    recommender = RecommenderService()
    
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    
    try:
        # Get recommendations
        recommendations = recommender.get_recommendations(
            user_id=input_data['user_id'],
            movie_ids=input_data['movie_ids'],
            top_n=input_data.get('top_n', 10)
        )
        
        # Write output to stdout
        print(json.dumps({
            'success': True,
            'recommendations': recommendations
        }))
        
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': str(e)
        }))
        sys.exit(1)

if __name__ == '__main__':
    main() 