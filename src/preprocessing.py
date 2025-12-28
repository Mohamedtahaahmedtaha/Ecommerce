import pandas as pd
import datetime as dt
import os
import numpy as np
from sklearn.ensemble import IsolationForest

def load_data(filepath):
    """
    Loads raw data from CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Using ISO-8859-1 encoding common for this dataset
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    return df

def clean_data(df):
    """
    Performs data cleaning: removes nulls, duplicates, cancellations,
    and invalid stock codes.
    """
    # Drop rows with missing crucial identifiers
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Remove cancelled transactions (InvoiceNo starts with 'C')
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    
    # Filter out invalid stock codes (non-numeric mostly)
    unique_stock_codes = df['StockCode'].unique()
    anomalous_stock_codes = [code for code in unique_stock_codes if sum(c.isdigit() for c in str(code)) in (0, 1)]
    df = df[~df['StockCode'].isin(anomalous_stock_codes)]
    
    # Filter out service-related descriptions
    service_related_descriptions = ["Next Day Carriage", "High Resolution Image"]
    df = df[~df['Description'].isin(service_related_descriptions)]
    
    # Standardize description and filter positive unit prices
    df['Description'] = df['Description'].str.upper()
    df = df[df['UnitPrice'] > 0]
    
    # Date conversion
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Calculate Total Spend per transaction line
    df['Total_Spend'] = df['UnitPrice'] * df['Quantity']
    
    return df

def create_rfm_features(df):
    """
    Aggregates data to Customer level using Recency, Frequency, and Monetary logic.
    """
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    # Aggregate data by CustomerID
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
        'InvoiceNo': 'nunique',                                 # Frequency
        'Total_Spend': 'sum'                                    # Monetary
    }).reset_index()
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'Total_Spend': 'Monetary'
    }, inplace=True)
    
    # Add Product Variety feature
    product_variety = df.groupby('CustomerID')['StockCode'].nunique().reset_index()
    product_variety.rename(columns={'StockCode': 'Unique_Products'}, inplace=True)
    
    rfm = pd.merge(rfm, product_variety, on='CustomerID')
    
    return rfm

def remove_outliers(df):
    """
    Uses Isolation Forest to remove outliers from the dataset.
    """
    model = IsolationForest(contamination=0.05, random_state=42)
    
    # Fit on the features
    features = ['Recency', 'Frequency', 'Monetary', 'Unique_Products']
    df['Outlier_Scores'] = model.fit_predict(df[features])
    
    # Keep only inliers (score == 1)
    clean_df = df[df['Outlier_Scores'] == 1].drop(columns=['Outlier_Scores'])
    
    return clean_df

if __name__ == "__main__":
    # Define paths dynamically relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to find the data folder
    base_dir = os.path.dirname(current_dir)
    
    # RAW PATH: C:\Users\LENOVO\Ecommerce production\data\data.csv
    # We use os.path.join for safety
    RAW_DATA_PATH = os.path.join(base_dir, 'data', 'data.csv')
    PROCESSED_DATA_PATH = os.path.join(base_dir, 'data', 'processed_data.csv')
    
    print(f"Looking for data at: {RAW_DATA_PATH}")
    
    # Pipeline execution
    try:
        raw_df = load_data(RAW_DATA_PATH)
        cleaned_df = clean_data(raw_df)
        rfm_df = create_rfm_features(cleaned_df)
        final_df = remove_outliers(rfm_df)
        
        # Save processed data
        final_df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Preprocessing complete. Processed data saved to {PROCESSED_DATA_PATH}")
        print(f"Final shape: {final_df.shape}")
        
    except FileNotFoundError:
        print("Error: Could not find the data file. Please check the path in lines 114-118.")