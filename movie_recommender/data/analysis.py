import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import os

# Set style for better visualizations
plt.style.use('seaborn-v0_8')  # Updated style name
sns.set_theme()  # Use seaborn's default theme

def load_data():
    """Load and merge the MovieLens datasets"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct full paths to the CSV files
    movies_path = os.path.join(script_dir, 'movies.csv')
    links_path = os.path.join(script_dir, 'links.csv')
    tags_path = os.path.join(script_dir, 'tags.csv')
    
    movies = pd.read_csv(movies_path)
    links = pd.read_csv(links_path)
    tags = pd.read_csv(tags_path)
    return movies, links, tags

def analyze_movies(movies):
    """Analyze movie dataset patterns"""
    print("\n=== Movie Analysis ===")
    
    # Basic statistics
    print(f"Total number of movies: {len(movies)}")
    
    # Genre analysis
    all_genres = [genre for genres in movies['genres'].str.split('|') for genre in genres]
    genre_counts = Counter(all_genres)
    print("\nTop 10 most common genres:")
    for genre, count in genre_counts.most_common(10):
        print(f"{genre}: {count} movies")
    
    # Year analysis
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    year_counts = movies['year'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    year_counts.plot(kind='line')
    plt.title('Number of Movies Released by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('movie_releases_by_year.png')
    
    # Genre distribution visualization
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(genre_counts.keys()), y=list(genre_counts.values()))
    plt.title('Genre Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('genre_distribution.png')

def analyze_links(links):
    """Analyze links dataset patterns"""
    print("\n=== Links Analysis ===")
    
    # Basic statistics
    print(f"Total number of links: {len(links)}")
    print(f"Number of movies with IMDB links: {links['imdbId'].notna().sum()}")
    print(f"Number of movies with TMDB links: {links['tmdbId'].notna().sum()}")
    
    # Missing data analysis
    missing_imdb = links['imdbId'].isna().sum()
    missing_tmdb = links['tmdbId'].isna().sum()
    print(f"\nMissing IMDB links: {missing_imdb} ({missing_imdb/len(links)*100:.2f}%)")
    print(f"Missing TMDB links: {missing_tmdb} ({missing_tmdb/len(links)*100:.2f}%)")

def analyze_tags(tags):
    """Analyze tags dataset patterns"""
    print("\n=== Tags Analysis ===")
    
    # Basic statistics
    print(f"Total number of tags: {len(tags)}")
    print(f"Number of unique users who created tags: {tags['userId'].nunique()}")
    print(f"Number of unique movies that were tagged: {tags['movieId'].nunique()}")
    
    # Tag frequency analysis
    tag_counts = tags['tag'].value_counts()
    print("\nTop 10 most common tags:")
    for tag, count in tag_counts.head(10).items():
        print(f"{tag}: {count} times")
    
    # Tags per movie analysis
    tags_per_movie = tags.groupby('movieId')['tag'].count()
    print(f"\nAverage tags per movie: {tags_per_movie.mean():.2f}")
    print(f"Maximum tags for a single movie: {tags_per_movie.max()}")
    
    # Tags per user analysis
    tags_per_user = tags.groupby('userId')['tag'].count()
    print(f"\nAverage tags per user: {tags_per_user.mean():.2f}")
    print(f"Maximum tags by a single user: {tags_per_user.max()}")

def main():
    """Main function to run all analyses"""
    print("Loading data...")
    movies, links, tags = load_data()
    
    analyze_movies(movies)
    analyze_links(links)
    analyze_tags(tags)
    
    print("\nAnalysis complete. Check the generated plots for visualizations.")

if __name__ == "__main__":
    main()
