from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(title="Customer Segmentation API", version="1.0")

# Define paths to models (these paths match the Docker volume mount)
# In Docker, we will mount the local 'models' folder to '/app/models'
MODEL_DIR = "/app/models"

# Global variables to hold loaded models
models = {}

@app.on_event("startup")
def load_models():
    """
    Load model artifacts on application startup.
    """
    try:
        # Check if running locally or in docker for debugging
        if not os.path.exists(MODEL_DIR):
             # Fallback for local testing if not using Docker
            print("Docker volume not found, checking local relative path...")
            local_model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
            models['scaler'] = joblib.load(os.path.join(local_model_path, "scaler.pkl"))
            models['pca'] = joblib.load(os.path.join(local_model_path, "pca.pkl"))
            models['kmeans'] = joblib.load(os.path.join(local_model_path, "kmeans_model.pkl"))
        else:
            models['scaler'] = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
            models['pca'] = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
            models['kmeans'] = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
            
        print("All models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print("Please ensure you have run 'src/train.py' first.")

# Define request body schema
class CustomerFeatures(BaseModel):
    Recency: int
    Frequency: int
    Monetary: float
    Unique_Products: int

@app.get("/")
def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "service": "Customer Segmentation API"}

@app.post("/predict")
def predict_segment(features: CustomerFeatures):
    """
    Predicts the customer segment based on RFM and Product features.
    """
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded")
        
    try:
        # Prepare input array
        input_data = np.array([[
            features.Recency,
            features.Frequency,
            features.Monetary,
            features.Unique_Products
        ]])
        
        # Apply preprocessing pipeline
        scaled_data = models['scaler'].transform(input_data)
        pca_data = models['pca'].transform(scaled_data)
        
        # Predict cluster
        cluster_id = int(models['kmeans'].predict(pca_data)[0])
        
        # Map cluster ID to business label
        segment_map = {
            0: "Lost / Low Value",
            1: "Loyal / High Value", 
            2: "Potential / Average"
        }
        
        return {
            "cluster_id": cluster_id,
            "segment_name": segment_map.get(cluster_id, "Unknown"),
            "input_features": features.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))