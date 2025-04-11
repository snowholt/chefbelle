import os
import pandas as pd
import sqlite3
from snippet import setup_multi_gpu_environment, setup_and_load_vector_databases, DB_PATH, VECTOR_DB_PATH

# Load the vectorized recipes dataframe from CSV/pickle
try:
    # Try to load from pickle for speed
    recipes_file = "datasets/vectorized_recipes_df.pkl"
    if os.path.exists(recipes_file):
        print(f"Loading vectorized recipes from {recipes_file}")
        vectorized_recipes_df = pd.read_pickle(recipes_file)
    else:
        # Fall back to CSV
        recipes_file = "datasets/vectorized_recipes_df.csv"
        print(f"Loading vectorized recipes from {recipes_file}")
        vectorized_recipes_df = pd.read_csv(recipes_file)
    
    print(f"✅ Loaded {len(vectorized_recipes_df)} vectorized recipes")
    print(f"Columns: {vectorized_recipes_df.columns.tolist()}")
    print("\nSample row:")
    print(vectorized_recipes_df.iloc[0])
    
    # Set up GPU acceleration
    gpu_available = setup_multi_gpu_environment()
    print(f"GPU acceleration enabled: {gpu_available}")
    
    # Connect to existing SQL database or create a new one
    print("\nConnecting to SQLite database...")
    sqlite_conn = sqlite3.connect(DB_PATH)
    
    # Load the vectorized data into ChromaDB
    print("\nSetting up vector database with tokenized data...")
    sqlite_conn, chroma_client, recipe_collection, user_collection = setup_and_load_vector_databases(
        vectorized_recipes_df=vectorized_recipes_df,
        sqlite_conn=sqlite_conn,
        db_path=DB_PATH,
        use_gpu=gpu_available
    )
    
    # Example similarity search query
    print("\n--- Example Similarity Search ---")
    query = "chicken pasta dinner"
    results = recipe_collection.query(
        query_texts=[query],
        n_results=5
    )
    
    print(f"\nSearch results for query: '{query}'")
    for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0],
            results['distances'][0])):
        print(f"\n{i+1}. Recipe ID: {metadata.get('recipe_id')}")
        print(f"   Similarity: {1 - distance:.4f}")
        print(f"   Calorie Level: {metadata.get('calorie_level', 'Unknown')}")
    
    print("\nVector database setup complete!")
    print("- SQLite database is available at:", DB_PATH)
    print("- Vector database is available at:", VECTOR_DB_PATH)
    
except Exception as e:
    print(f"Error loading or processing data: {e}")
