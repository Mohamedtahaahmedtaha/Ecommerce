import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def train_model():
    """
    Trains the clustering pipeline and saves artifacts.
    """
    # Define paths dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    
    DATA_PATH = os.path.join(base_dir, 'data', 'processed_data.csv')
    MODELS_DIR = os.path.join(base_dir, 'models')
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    print("Loading processed data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Processed data not found. Run preprocessing.py first.")
        
    df = pd.read_csv(DATA_PATH)
    
    # Drop CustomerID for training
    X = df.drop(columns=['CustomerID'])
    
    # 1. Scaling
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. PCA (Dimensionality Reduction)
    # Using 6 components based on analysis
    print("Applying PCA...")
    pca = PCA(n_components=3) 
    X_pca = pca.fit_transform(X_scaled)
    
    # 3. K-Means Clustering
    # Using 3 clusters based on Elbow method analysis
    print("Training K-Means model...")
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=100, random_state=42)
    kmeans.fit(X_pca)
    
    # Save Artifacts
    print("Saving model artifacts...")
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(pca, os.path.join(MODELS_DIR, 'pca.pkl'))
    joblib.dump(kmeans, os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    
    print(f"Training pipeline completed successfully. Models saved to {MODELS_DIR}")

if __name__ == "__main__":
    train_model()