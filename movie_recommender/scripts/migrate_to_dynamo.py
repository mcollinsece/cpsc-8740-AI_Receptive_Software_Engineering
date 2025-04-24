import boto3
import pandas as pd
import json
import time
from tqdm import tqdm
import os
from typing import Dict, List, Any
from decimal import Decimal

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb')

def get_table(table_name: str):
    """Get DynamoDB table by name"""
    return dynamodb.Table(table_name)

def batch_write_items(table, items: List[Dict[str, Any]], batch_size: int = 25):
    """Write items to DynamoDB in batches"""
    with table.batch_writer() as batch:
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            for item in batch_items:
                batch.put_item(Item=item)

def migrate_movies():
    """Migrate movies data"""
    print("Migrating movies data...")
    movies_df = pd.read_csv('../data/movies.csv')
    table = get_table('movie-recommender-movies')
    
    items = []
    for _, row in tqdm(movies_df.iterrows(), total=len(movies_df)):
        item = {
            'movieId': int(row['movieId']),
            'title': row['title'],
            'genres': row['genres']
        }
        items.append(item)
    
    batch_write_items(table, items)
    print(f"Migrated {len(items)} movies")

def migrate_links():
    """Migrate links data"""
    print("Migrating links data...")
    links_df = pd.read_csv('../data/links.csv')
    table = get_table('movie-recommender-links')
    
    items = []
    for _, row in tqdm(links_df.iterrows(), total=len(links_df)):
        item = {
            'movieId': int(row['movieId']),
            'imdbId': str(row['imdbId']),
            'tmdbId': str(row['tmdbId'])
        }
        items.append(item)
    
    batch_write_items(table, items)
    print(f"Migrated {len(items)} links")

def migrate_tags():
    """Migrate tags data"""
    print("Migrating tags data...")
    tags_df = pd.read_csv('../data/tags.csv')
    table = get_table('movie-recommender-tags')
    
    items = []
    for _, row in tqdm(tags_df.iterrows(), total=len(tags_df)):
        item = {
            'movieId': int(row['movieId']),
            'userId': int(row['userId']),
            'timestamp': int(row['timestamp']),  # Use timestamp as part of the key
            'tag': row['tag']
        }
        items.append(item)
    
    # Write items in smaller batches to avoid duplicates
    batch_size = 25
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        batch_write_items(table, batch_items)
    
    print(f"Migrated {len(items)} tags")

def migrate_ratings():
    """Migrate ratings data"""
    print("Migrating ratings data...")
    ratings_df = pd.read_csv('../data/ratings.csv')
    table = get_table('movie-recommender-ratings')
    
    items = []
    for _, row in tqdm(ratings_df.iterrows(), total=len(ratings_df)):
        item = {
            'userId': int(row['userId']),
            'movieId': int(row['movieId']),
            'rating': Decimal(str(row['rating'])),  # Convert float to Decimal
            'timestamp': int(row['timestamp'])
        }
        items.append(item)
    
    batch_write_items(table, items)
    print(f"Migrated {len(items)} ratings")

def main():
    print("Starting data migration to DynamoDB...")
    
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    # Check if CSV files exist
    required_files = ['movies.csv', 'links.csv', 'tags.csv', 'ratings.csv']
    for file in required_files:
        file_path = os.path.join('../data', file)
        if not os.path.exists(file_path):
            print(f"Error: Required file {file} not found in {os.path.abspath('../data')}")
            return
    
    # Migrate data in order of dependencies
    #migrate_movies()
    #migrate_links()
    migrate_tags()
    #migrate_ratings()
    
    print("Data migration completed successfully!")

if __name__ == "__main__":
    main() 