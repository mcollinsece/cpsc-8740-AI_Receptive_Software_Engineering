import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import json
from models.recommender import Recommender

# Setting the device
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

class RatingsDataset(Dataset):
    def __init__(self, data):
        self.data = data.nonzero(as_tuple=True)
        self.ratings = data[self.data]

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        user = self.data[0][idx]
        item = self.data[1][idx]
        rating = self.ratings[idx]
        return user, item, rating

def train_model():
    # Load and prepare data
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')

    # Create mappings
    movie_ids = list(movies.movieId)
    moviesid_to_title = dict(zip(movies.movieId, movies.title))
    
    # Create user-rating matrix
    user_rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Create movie ID mappings
    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(user_rating_matrix.columns)}
    idx_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_idx.items()}
    
    # Save mappings for inference
    mappings = {
        'movie_id_to_idx': movie_id_to_idx,
        'idx_to_movie_id': {str(k): v for k, v in idx_to_movie_id.items()},  # Convert keys to strings for JSON
        'moviesid_to_title': moviesid_to_title
    }
    with open('model_mappings.json', 'w') as f:
        json.dump(mappings, f)

    # Prepare training data
    user_rating_matrix_np = user_rating_matrix.values
    train_data, test_data = train_test_split(user_rating_matrix_np, test_size=0.2, random_state=42)
    train_data = torch.FloatTensor(train_data)
    test_data = torch.FloatTensor(test_data)

    # Create data loaders
    batch_size = 64
    train_dataset = RatingsDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = RatingsDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    num_users, num_items = user_rating_matrix_np.shape
    model = Recommender(num_users, num_items).to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=2e-5)
    epochs = 4

    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for user, item, rating in train_loader:
            user, item, rating = user.to(device), item.to(device), rating.to(device)
            optimizer.zero_grad()
            output = model(user, item).squeeze()
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader)}")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_users': num_users,
        'num_items': num_items
    }, 'recommender_model.pth')

if __name__ == '__main__':
    train_model() 