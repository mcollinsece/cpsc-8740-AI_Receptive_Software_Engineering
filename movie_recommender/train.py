import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import json
from models.recommender import Recommender
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

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

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for user, item, rating in data_loader:
            user, item, rating = user.to(device), item.to(device), rating.to(device)
            output = model(user, item).squeeze()
            loss = criterion(output, rating)
            total_loss += loss.item()
            
            predictions.extend(output.cpu().numpy())
            actuals.extend(rating.cpu().numpy())
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    
    return {
        'loss': total_loss / len(data_loader),
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'predictions': predictions,
        'actuals': actuals
    }

def plot_training_history(train_metrics, val_metrics):
    plt.figure(figsize=(12, 4))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_metrics['loss'], label='Training Loss')
    plt.plot(val_metrics['loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot RMSE and MAE
    plt.subplot(1, 2, 2)
    plt.plot(val_metrics['rmse'], label='RMSE')
    plt.plot(val_metrics['mae'], label='MAE')
    plt.title('Error Metrics Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_prediction_analysis(final_metrics):
    plt.figure(figsize=(12, 4))
    
    # Scatter plot of predicted vs actual ratings
    plt.subplot(1, 2, 1)
    plt.scatter(final_metrics['actuals'], final_metrics['predictions'], alpha=0.5)
    plt.plot([0, 5], [0, 5], 'r--')  # Perfect prediction line
    plt.title('Predicted vs Actual Ratings')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    
    # Distribution of prediction errors
    plt.subplot(1, 2, 2)
    errors = np.array(final_metrics['predictions']) - np.array(final_metrics['actuals'])
    sns.histplot(errors, bins=50)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png')
    plt.close()

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
        'idx_to_movie_id': {str(k): v for k, v in idx_to_movie_id.items()},
        'moviesid_to_title': moviesid_to_title
    }
    with open('model_mappings.json', 'w') as f:
        json.dump(mappings, f)

    # Prepare training data
    user_rating_matrix_np = user_rating_matrix.values
    train_data, val_data = train_test_split(user_rating_matrix_np, test_size=0.2, random_state=42)
    train_data = torch.FloatTensor(train_data)
    val_data = torch.FloatTensor(val_data)

    # Create data loaders
    batch_size = 64
    train_dataset = RatingsDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = RatingsDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    num_users, num_items = user_rating_matrix_np.shape
    model = Recommender(num_users, num_items).to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=2e-5)
    epochs = 4

    # Metrics tracking
    train_metrics = {'loss': [], 'mse': [], 'rmse': [], 'mae': []}
    val_metrics = {'loss': [], 'mse': [], 'rmse': [], 'mae': []}

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for user, item, rating in train_loader:
            user, item, rating = user.to(device), item.to(device), rating.to(device)
            optimizer.zero_grad()
            output = model(user, item).squeeze()
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Evaluate on training set
        train_results = evaluate_model(model, train_loader, criterion, device)
        val_results = evaluate_model(model, val_loader, criterion, device)
        
        # Store metrics
        for metric in ['loss', 'mse', 'rmse', 'mae']:
            train_metrics[metric].append(train_results[metric])
            val_metrics[metric].append(val_results[metric])
        
        print(f"Epoch {epoch}")
        print(f"Training - Loss: {train_results['loss']:.4f}, RMSE: {train_results['rmse']:.4f}, MAE: {train_results['mae']:.4f}")
        print(f"Validation - Loss: {val_results['loss']:.4f}, RMSE: {val_results['rmse']:.4f}, MAE: {val_results['mae']:.4f}")
        print("-" * 50)

    # Plot training history
    plot_training_history(train_metrics, val_metrics)
    
    # Final evaluation and prediction analysis
    final_metrics = evaluate_model(model, val_loader, criterion, device)
    plot_prediction_analysis(final_metrics)

    # Save model and metrics
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_users': num_users,
        'num_items': num_items,
        'final_metrics': {
            'rmse': final_metrics['rmse'],
            'mae': final_metrics['mae']
        }
    }, 'recommender_model.pth')

    # Save metrics summary
    metrics_summary = {
        'final_rmse': final_metrics['rmse'],
        'final_mae': final_metrics['mae'],
        'training_history': {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
    }
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics_summary, f)

if __name__ == '__main__':
    train_model() 