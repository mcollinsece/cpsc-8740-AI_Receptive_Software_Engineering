# Movie Recommender System

A comprehensive movie recommendation system implementing various AI techniques for a graduate-level AI course.

## Features

- Collaborative Filtering (User-based and Item-based)
- Content-based Filtering
- Hybrid Recommendation System
- Web Interface for User Interaction
- Evaluation Metrics and Performance Analysis

## Project Structure

```
movie_recommender/
├── data/                  # Data files and processing scripts
├── models/               # Recommendation algorithms
├── web/                  # Web interface
├── evaluation/           # Evaluation metrics and analysis
└── utils/                # Utility functions
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the web application:
```bash
python web/app.py
```

## Data

The system uses the MovieLens dataset, which includes:
- Movie ratings
- Movie metadata
- User information

## Algorithms

1. Collaborative Filtering
   - User-based collaborative filtering
   - Item-based collaborative filtering
   - Matrix factorization (SVD)

2. Content-based Filtering
   - TF-IDF based movie similarity
   - Genre-based recommendations

3. Hybrid Approach
   - Weighted combination of collaborative and content-based filtering

## Evaluation

The system includes various evaluation metrics:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Precision@K
- Recall@K
- F1 Score 