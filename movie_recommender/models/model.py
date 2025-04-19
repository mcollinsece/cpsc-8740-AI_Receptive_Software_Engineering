# Importing initial packages
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette("deep")
import matplotlib.pyplot as plt

# Setting the device to GPU for parallelization
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

from flask import Flask, request, jsonify  # Optional: for deployment

# The dataset containing the movies
movies = pd.read_csv('data/movies.csv')
# The dataset containing the ratings by users
ratings = pd.read_csv('data/ratings.csv')

print(f"""The shape of movies is: {movies.shape}
The shape of rating is: {ratings.shape}""")


print(f"""The columns of movies is: {movies.columns.to_list()}
The columns of rating is: {ratings.columns.to_list()}""")

## The shape of movies is: (9742, 3)
## The shape of rating is: (100836, 4)
## The columns of movies is: ['movieId', 'title', 'genres']
## The columns of rating is: ['userId', 'movieId', 'rating', 'timestamp']

# Creating encoders and decoders for our movie ids
movie_ids = list(movies.movieId)                              # All Movie IDs
moviesid_to_title = dict(zip(movies.movieId,movies.title))    # Decoder
movietitle_to_id = {j:i for i,j in moviesid_to_title.items()} # Encoder

# NAs are not watched by the user, so rating of 0
# Creates a matrix where each row represents a user and each column represents a movie
# The values are the ratings (0-5) that each user gave to each movie
# If a user hasn't rated a movie (NaN), it's filled with 0
user_rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
print(user_rating_matrix.head())

# Create mapping from movie ID to embedding index
movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(user_rating_matrix.columns)}
idx_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_idx.items()}

# Convert this from a pandas df to numpy array
user_rating_matrix_np = user_rating_matrix.values
# Check out the shape of our new array
print(f"Shape of the numpy array: {user_rating_matrix_np.shape}; represting {user_rating_matrix_np.shape[0]} users with {user_rating_matrix_np.shape[1]} movies.")

## Shape of the numpy array: (610, 9724); represting 610 users with 9724 movies.


# Split the full dataset into into training and test sets
train_data, test_data = train_test_split(user_rating_matrix_np, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_data = torch.FloatTensor(train_data)
test_data = torch.FloatTensor(test_data)


class RatingsDataset(Dataset):
    def __init__(self, data):
        self.data = data.nonzero(as_tuple=True) # Get the indices of non-zero elements
        self.ratings = data[self.data]          # Using non-zero indiced to extract those ratings

    def __len__ (self):
        return len(self.data[0])
    
    def __getitem__ (self, idx):
        user = self.data[0][idx]   # Extracting the user
        item = self.data[1][idx]   # Extracting the movie
        rating = self.ratings[idx] # Extracting the rating of the movie for that user
        
        return user, item, rating
 
# Setting batch size: the amount of examples handled at once
batch_size = 64 

# Creating the Training DataLoader
train_dataset = RatingsDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
# Creating the Test DataLoader
test_dataset = RatingsDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

print(f"Number of samples in train_dataset: {len(train_dataset)}")
print(f"Number of samples in test_dataset: {len(test_dataset)}")

## Number of samples in train_dataset: 73177
## Number of samples in test_dataset: 27659


# Model definition
class Recommender(nn.Module):
    def __init__(self, num_users, num_items, n_embd = 16):
        super(Recommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, n_embd) # Creating an embedding for our users with the correct dimensions
        self.item_embedding = nn.Embedding(num_items, n_embd) # Embedding items
        
        self.fc1 = nn.Linear(n_embd * 2, 32)                  # First fully connected (fc) layer
        self.fc2 = nn.Linear(32, 16)                          # Second fc layer
        self.fc3 = nn.Linear(16, 8)                           # Third fc layer
        self.fc4 = nn.Linear(8, 1)                            # Final fc layer, with an output of one value
       
        self.sigmoid = nn.Sigmoid()                           # Sigmoid activation function to compress the output to a value between (0,1)
        
        self.dropout1 = nn.Dropout(0.2)                       # Dropout layer to prevent overfitting
        self.dropout2 = nn.Dropout(0.5)                       # Additional Dropout layer to prevent overfitting
        
        self.bn1 = nn.BatchNorm1d(32)                         # Batch Normalization to reduce skew
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(8)


    def forward(self, user, item):
        user_embed = self.user_embedding(user)          # Embedding our users
        item_embed = self.item_embedding(item)          # Embedding the movies
        
        x = torch.cat([user_embed, item_embed], dim=-1) # Concatenating the users and items 
        x = torch.relu(self.fc1(x))                     # Applying first fc layer, with a ReLU activation function
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.fc4(x)                                 # Final layer to get predicted rating
        x = self.sigmoid(x)                             # Applying the sigmoid function
        x = x*5                                         # Scale the output from [0,1] to [0,5]   
        return x


# Defining model parameters (amount of users and movies) based on our data
num_users, num_items = user_rating_matrix_np.shape

# Initialising our model
model = Recommender(num_users, num_items)
# Setting our model to the device (if applicable)
model = model.to(device)

# Mean Squared Error
criterion = nn.MSELoss()
# Adaptive Moment Estimation, the most common optimizer due to efficiency and effectiveness
# learning rate (lr) controls how much to adjust the model's parameters with respect to loss gradient
optimizer = optim.Adam(model.parameters(), lr = 0.003, weight_decay=2e-5)

epochs = 4    # Epochs are the amount of time's to loop over the training data
model.train()
loss_dict = {}
for epoch in range(1, epochs+1):
    total_loss = 0
    for user, item, rating in train_loader:
         user, item, rating = user.to(device), item.to(device), rating.to(device)
         optimizer.zero_grad()
         output = model(user, item).squeeze()
         loss = criterion(output, rating)
         loss.backward()
         optimizer.step()
         total_loss += loss.item()
      
    # Print loss after every epoch
    epoch_loss = total_loss/len(train_loader)      # Get the average loss for this epoch
    loss_dict[epoch] = epoch_loss                  # Store this epoch, loss into a dict for graphing
    print(f"Epoch {epoch}, Loss: {epoch_loss}")    # Print the average loss for this epoch

## Epoch 1, Loss: 1.137577333441981
## Epoch 2, Loss: 0.9602082769770722
## Epoch 3, Loss: 0.8480213888354234
## Epoch 4, Loss: 0.7838263336855632

model.eval()
total_loss = 0
with torch.no_grad():
    for user, item, rating in test_loader:
        user, item, rating = user.to(device), item.to(device), rating.to(device)
        output = model(user, item).squeeze()
        loss = criterion(output, rating)
        total_loss += loss.item()
    
test_loss = total_loss / len(test_loader)   

print(f"Test Loss: {test_loss}")
## Test Loss: 0.8564968511303908

def recommend_movies(user_id, movie_ids, top_n=10):
    """This function gets the recommendations for all movies for a user, and returns the top_n"""
    model.eval()                                 # Set model to evaluation mode
    
    # Convert movie IDs to embedding indices
    valid_movie_indices = [movie_id_to_idx[mid] for mid in movie_ids if mid in movie_id_to_idx]
    if not valid_movie_indices:
        raise ValueError("No valid movie IDs found in the training data")
    
    user = torch.tensor([user_id] * len(valid_movie_indices)).to(device)  # Ensure tensor is on the same device as the model
    movies = torch.tensor(valid_movie_indices).to(device)  # Use indices instead of raw movie IDs
    with torch.no_grad():
        predicted_ratings = model(user, movies).squeeze() # Running the model, getting correct dimensions
    top_movie_indices = predicted_ratings.argsort(descending=True)[:top_n] # Sorting the ratings in order, then slicing at top_n
    recommended_movie_ids = [idx_to_movie_id[idx.item()] for idx in movies[top_movie_indices]]
    recommended_ratings = predicted_ratings[top_movie_indices].tolist()
    recommendations = list(zip([user_id] * len(recommended_movie_ids), recommended_movie_ids, recommended_ratings))
    return recommendations

# Example usage for recommendations
user_id = 5 
top_n = 1   # Only retrieving the top 1 movie for this user
recommended_movies = recommend_movies(user_id, list(user_rating_matrix.columns), top_n)  # Use only valid movie IDs from training data

print(f"For User: {recommended_movies[0][0]}, \"{moviesid_to_title[recommended_movies[0][1]]}\" has an estimated rating of {recommended_movies[0][2]:.2f}")

## For User: 5, "Miracle on 34th Street (1994)" has an estimated rating of 4.32


app = Flask(__name__)                       # Initialize a Flask application

@app.route('/recommend', methods=['POST'])  # Define a POST endpoint for recommendations
def recommend():
    user_id = request.json['user_id']       # Get the user ID from the request JSON
    item_ids = request.json['item_ids']     # Get the list of item IDs from the request JSON
    user = torch.LongTensor([user_id] * len(item_ids))  # Create a tensor of user IDs (same user repeated)
    items = torch.LongTensor(item_ids)      # Create a tensor of item IDs
    with torch.no_grad():                   # Disable gradient computation for inference
        predictions = model(user, items).squeeze().tolist()  # Get the model's predictions and convert to list
    recommendations = {item_id: pred for item_id, pred in zip(item_ids, predictions)}  # Create a dictionary of item IDs and their predicted ratings
    return jsonify(recommendations)         # Return the recommendations as a JSON response

if __name__ == '__main__':
    app.run(debug=True)                     # Run the Flask application in debug mode