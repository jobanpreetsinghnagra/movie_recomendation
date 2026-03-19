"""
Movie Recommendation System using PyTorch Matrix Factorization
Implements collaborative filtering with 80-20 train-test split
Recommends top 3 movies for any user based on their preferences
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PYTORCH MODEL: Matrix Factorization with Biases
# ============================================================================

class MatrixFactorizationWithBias(nn.Module):
    """
    Matrix Factorization model with user and item biases.
    Learns latent factors to predict user-item ratings.
    """
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(MatrixFactorizationWithBias, self).__init__()
        
        # User and item embeddings (latent factors)
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # User and item biases
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # Global bias (overall average)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights for better convergence
        self.user_embeddings.weight.data.uniform_(0, 0.05)
        self.item_embeddings.weight.data.uniform_(0, 0.05)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)
    
    def forward(self, user_ids, item_ids):
        """
        Forward pass: compute predicted ratings
        
        Args:
            user_ids: Tensor of user indices
            item_ids: Tensor of item indices
            
        Returns:
            Predicted ratings
        """
        # Get latent vectors
        user_vecs = self.user_embeddings(user_ids)
        item_vecs = self.item_embeddings(item_ids)
        
        # Dot product of latent factors
        dot_product = (user_vecs * item_vecs).sum(dim=1)
        
        # Add biases
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        
        # Final prediction: dot product + biases + global bias
        prediction = dot_product + user_b + item_b + self.global_bias
        
        return prediction


# ============================================================================
# DATASET CLASS
# ============================================================================

class MovieRatingsDataset(Dataset):
    """Custom PyTorch Dataset for movie ratings"""
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids.values)
        self.item_ids = torch.LongTensor(item_ids.values)
        self.ratings = torch.FloatTensor(ratings.values)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


# ============================================================================
# MOVIE RECOMMENDER CLASS
# ============================================================================

class MovieRecommender:
    """
    Complete recommendation system that handles:
    - Data loading and preprocessing
    - Model training with 80-20 split
    - Top-3 movie recommendations
    """
    
    def __init__(self, ratings_path, movies_path=None, embedding_dim=50):
        """
        Initialize the recommender system
        
        Args:
            ratings_path: Path to ratings.csv (userId, movieId, rating, timestamp)
            movies_path: Path to movies_encoded.csv (optional, for movie metadata)
            embedding_dim: Size of latent feature vectors
        """
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("=" * 80)
        print("INITIALIZING MOVIE RECOMMENDATION SYSTEM")
        print("=" * 80)
        
        # Load ratings data
        print(f"\nLoading ratings from: {ratings_path}")
        self.ratings_df = pd.read_csv(ratings_path)
        print(f"✓ Loaded {len(self.ratings_df):,} ratings")
        
        # Load movie information if available
        self.has_movie_info = False
        if movies_path:
            try:
                self.movies_df = pd.read_csv(movies_path)
                self.has_movie_info = True
                print(f"✓ Loaded {len(self.movies_df):,} movies with metadata")
            except FileNotFoundError:
                print(f"⚠ Could not load {movies_path}, will use movie IDs only")
        
        # Create user and item mappings
        self._create_mappings()
        
        # Initialize model
        self.model = None
        self.train_losses = []
        self.test_losses = []
        
    def _create_mappings(self):
        """Create mappings between original IDs and model indices"""
        # Get unique users and items
        unique_users = sorted(self.ratings_df['userId'].unique())
        unique_items = sorted(self.ratings_df['movieId'].unique())
        
        # Create bidirectional mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Add mapped columns to dataframe
        self.ratings_df['user_idx'] = self.ratings_df['userId'].map(self.user_to_idx)
        self.ratings_df['item_idx'] = self.ratings_df['movieId'].map(self.item_to_idx)
        
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   • Users: {self.num_users:,}")
        print(f"   • Movies: {self.num_items:,}")
        print(f"   • Ratings: {len(self.ratings_df):,}")
        print(f"   • Sparsity: {100 * (1 - len(self.ratings_df) / (self.num_users * self.num_items)):.2f}%")
    
    def prepare_data(self, test_size=0.2, batch_size=128, random_state=42):
        """
        Split data into 80-20 train-test and create DataLoaders
        
        Args:
            test_size: Fraction for testing (0.2 = 20%)
            batch_size: Batch size for training
            random_state: Random seed for reproducibility
        """
        # 80-20 train-test split
        train_df, test_df = train_test_split(
            self.ratings_df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        print(f"\n📂 Data Split (80-20):")
        print(f"   • Training set: {len(train_df):,} ratings ({100*(1-test_size):.0f}%)")
        print(f"   • Test set: {len(test_df):,} ratings ({100*test_size:.0f}%)")
        
        # Create PyTorch datasets
        train_dataset = MovieRatingsDataset(
            train_df['user_idx'],
            train_df['item_idx'],
            train_df['rating']
        )
        
        test_dataset = MovieRatingsDataset(
            test_df['user_idx'],
            test_df['item_idx'],
            test_df['rating']
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.train_df = train_df
        self.test_df = test_df
        
        # Store rating statistics
        self.mean_rating = train_df['rating'].mean()
        self.std_rating = train_df['rating'].std()
        
        print(f"   • Batch size: {batch_size}")
        print(f"   • Mean rating: {self.mean_rating:.2f} ± {self.std_rating:.2f}")
    
    def build_model(self):
        """Initialize the matrix factorization model"""
        self.model = MatrixFactorizationWithBias(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        # Set global bias to mean rating
        self.model.global_bias.data.fill_(self.mean_rating)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n🧠 Model Architecture:")
        print(f"   • Embedding dimension: {self.embedding_dim}")
        print(f"   • Total parameters: {total_params:,}")
        print(f"   • Device: {self.device}")
    
    def train(self, epochs=20, lr=0.01, weight_decay=1e-5, verbose=True):
        """
        Train the recommendation model
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization
            verbose: Print training progress
        """
        if self.model is None:
            self.build_model()
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        print(f"\n🚀 Training started...")
        print("=" * 80)
        
        best_test_loss = float('inf')
        
        for epoch in range(epochs):
            # TRAINING PHASE
            self.model.train()
            train_loss = 0.0
            
            for user_ids, item_ids, ratings in self.train_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                # Forward pass
                predictions = self.model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            
            # EVALUATION PHASE
            self.model.eval()
            test_loss = 0.0
            
            with torch.no_grad():
                for user_ids, item_ids, ratings in self.test_loader:
                    user_ids = user_ids.to(self.device)
                    item_ids = item_ids.to(self.device)
                    ratings = ratings.to(self.device)
                    
                    predictions = self.model(user_ids, item_ids)
                    loss = criterion(predictions, ratings)
                    test_loss += loss.item()
            
            test_loss /= len(self.test_loader)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            # Track best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch + 1
            
            # Print progress
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"Epoch [{epoch+1:2d}/{epochs}] | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Test Loss: {test_loss:.4f} | "
                      f"RMSE: {np.sqrt(test_loss):.4f}")
        
        print("=" * 80)
        print(f"✓ Training complete!")
        print(f"   • Best test RMSE: {np.sqrt(best_test_loss):.4f} (epoch {best_epoch})")
        print(f"   • Final train RMSE: {np.sqrt(self.train_losses[-1]):.4f}")
        print(f"   • Final test RMSE: {np.sqrt(self.test_losses[-1]):.4f}")
    
    def predict(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair
        
        Args:
            user_id: Original user ID
            movie_id: Original movie ID
            
        Returns:
            Predicted rating (or None if user/movie not in training data)
        """
        if user_id not in self.user_to_idx or movie_id not in self.item_to_idx:
            return None
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[movie_id]
        
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            item_tensor = torch.LongTensor([item_idx]).to(self.device)
            prediction = self.model(user_tensor, item_tensor)
            
        return prediction.item()
    
    def recommend_top_3(self, user_id, exclude_rated=True):
        """
        Get top 3 movie recommendations for a user
        
        Args:
            user_id: Original user ID
            exclude_rated: Whether to exclude movies user has already rated
            
        Returns:
            DataFrame with top 3 recommendations
        """
        if user_id not in self.user_to_idx:
            print(f"❌ User {user_id} not found in training data")
            return None
        
        user_idx = self.user_to_idx[user_id]
        
        # Get movies already rated by the user
        if exclude_rated:
            rated_movies = set(
                self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].values
            )
        else:
            rated_movies = set()
        
        # Get all available movies
        all_movie_ids = list(self.item_to_idx.keys())
        
        # Predict ratings for all unrated movies
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            # Batch prediction for efficiency
            unrated_movies = [m for m in all_movie_ids if m not in rated_movies]
            
            for movie_id in unrated_movies:
                movie_idx = self.item_to_idx[movie_id]
                
                user_tensor = torch.LongTensor([user_idx]).to(self.device)
                item_tensor = torch.LongTensor([movie_idx]).to(self.device)
                
                pred_rating = self.model(user_tensor, item_tensor).item()
                predictions.append((movie_id, pred_rating))
        
        # Sort by predicted rating (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3
        top_3 = predictions[:3]
        
        # Create result DataFrame
        results = []
        for rank, (movie_id, pred_rating) in enumerate(top_3, 1):
            result = {
                'rank': rank,
                'movieId': int(movie_id),
                'predicted_rating': round(pred_rating, 2)
            }
            
            # Add movie metadata if available
            if self.has_movie_info:
                movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
                if not movie_info.empty:
                    # Add available columns
                    for col in self.movies_df.columns:
                        if col != 'movieId' and col in movie_info.columns:
                            result[col] = movie_info[col].values[0]
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_user_profile(self, user_id):
        """Get user's rating statistics and profile"""
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return None
        
        profile = {
            'user_id': user_id,
            'num_ratings': len(user_ratings),
            'mean_rating': user_ratings['rating'].mean(),
            'min_rating': user_ratings['rating'].min(),
            'max_rating': user_ratings['rating'].max(),
            'std_rating': user_ratings['rating'].std()
        }
        
        return profile
    
    def save_model(self, path='movie_recommender_model.pth'):
        """Save trained model and mappings"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'embedding_dim': self.embedding_dim,
            'mean_rating': self.mean_rating,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
        }, path)
        print(f"\n💾 Model saved to: {path}")
    
    def load_model(self, path='movie_recommender_model.pth'):
        """Load a pre-trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.user_to_idx = checkpoint['user_to_idx']
        self.item_to_idx = checkpoint['item_to_idx']
        self.num_users = checkpoint['num_users']
        self.num_items = checkpoint['num_items']
        self.embedding_dim = checkpoint['embedding_dim']
        self.mean_rating = checkpoint['mean_rating']
        
        # Rebuild model
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ Model loaded from: {path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function demonstrating the complete recommendation workflow
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "PYTORCH MOVIE RECOMMENDATION SYSTEM")
    print("=" * 80)
    
    # File paths - UPDATE THESE TO YOUR CSV FILE LOCATIONS
    RATINGS_PATH = 'dataset/ratings.csv'
    MOVIES_PATH = 'dataset/movies_encoded.csv'  # Optional
    
    # Initialize recommender
    recommender = MovieRecommender(
        ratings_path=RATINGS_PATH,
        movies_path=MOVIES_PATH,
        embedding_dim=50
    )
    
    # Prepare data with 80-20 split
    recommender.prepare_data(test_size=0.2, batch_size=128, random_state=42)
    
    # Train the model
    recommender.train(epochs=20, lr=0.01, weight_decay=1e-5)
    
    # Save the trained model
    recommender.save_model('movie_recommender_model.pth')
    
    # DEMONSTRATION: Get recommendations for sample users
    print("\n" + "=" * 80)
    print("GENERATING TOP-3 RECOMMENDATIONS")
    print("=" * 80)
    
    # Get sample users
    sample_users = recommender.ratings_df['userId'].unique()[:3]
    
    for user_id in sample_users:
        print(f"\n{'─' * 80}")
        
        # Show user profile
        profile = recommender.get_user_profile(user_id)
        print(f"👤 USER {user_id} PROFILE:")
        print(f"   • Total ratings: {profile['num_ratings']}")
        print(f"   • Average rating: {profile['mean_rating']:.2f}")
        print(f"   • Rating range: [{profile['min_rating']:.1f} - {profile['max_rating']:.1f}]")
        
        # Get top 3 recommendations
        print(f"\n🎬 TOP 3 RECOMMENDATIONS:")
        recommendations = recommender.recommend_top_3(user_id, exclude_rated=True)
        
        if recommendations is not None:
            for _, row in recommendations.iterrows():
                print(f"   #{row['rank']}. Movie {row['movieId']} - "
                      f"Predicted Rating: {row['predicted_rating']:.2f}")
    
    # INTERACTIVE MODE
    print("\n" + "=" * 80)
    print("INTERACTIVE RECOMMENDATION MODE")
    print("=" * 80)
    print("\n💡 Enter a user ID to get personalized recommendations")
    print("   Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("Enter User ID: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Movie Recommender!")
                break
            
            user_id = int(user_input)
            
            # Check if user exists
            profile = recommender.get_user_profile(user_id)
            if profile is None:
                print(f"❌ User {user_id} not found. Try another ID.\n")
                continue
            
            # Show profile
            print(f"\n👤 USER {user_id}:")
            print(f"   Rated {profile['num_ratings']} movies (avg: {profile['mean_rating']:.2f})")
            
            # Get recommendations
            print(f"\n🎬 TOP 3 MOVIE RECOMMENDATIONS:\n")
            recommendations = recommender.recommend_top_3(user_id)
            
            if recommendations is not None:
                print(recommendations.to_string(index=False))
            print()
            
        except ValueError:
            print("⚠ Please enter a valid number or 'quit'\n")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()