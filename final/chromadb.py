# Import required libraries
import os
import json
import numpy as np
import pandas as pd
import sqlite3
import chromadb
from typing import Dict, List, Any, Optional, Tuple, Union
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm  # For progress bars

# Import embedding functions based on ChromaDB version
try:
    # For newer versions of ChromaDB
    from chromadb.utils import embedding_functions
except ImportError:
    try:
        # For older versions of ChromaDB
        from chromadb import embedding_functions
    except ImportError:
        print("Warning: Could not import embedding_functions from ChromaDB. Using default embedding function.")

print("# Step 3: Recipe Customization with Function Calling and Hybrid Storage")
print("## Setting up the hybrid database architecture")



print("# Step 3: Recipe Customization with Function Calling and Hybrid Storage")
print("## Setting up the hybrid database architecture")

# Define paths and connection variables
DB_PATH = "kitchen_db.sqlite"
VECTOR_DB_PATH = "vector_db"  # Directory where ChromaDB will store its persistent data

# Create directories if they don't exist
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# ============================================================
# PART 1.1: Hybrid Database Setup
# ============================================================

def setup_sqlite_database(db_path: str = DB_PATH):
    """Set up SQLite database with the required schema."""
    print("Setting up SQLite database...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create recipes table with updated columns to match dataset
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS recipes (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        minutes INTEGER,
        contributor_id INTEGER,
        submitted TEXT,
        tags TEXT,            -- JSON array of tags
        nutrition TEXT,       -- JSON object with nutrition info
        n_steps INTEGER,
        steps TEXT,           -- JSON array of steps
        description TEXT,
        ingredients TEXT,     -- JSON array of ingredients
        n_ingredients INTEGER,
        cuisine_type TEXT,
        dietary_tags TEXT     -- JSON array of dietary tags
    )
    ''')
    
    # Create interactions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        user_id INTEGER,
        recipe_id INTEGER,
        date TEXT,
        rating INTEGER,
        review TEXT,
        PRIMARY KEY (user_id, recipe_id),
        FOREIGN KEY (recipe_id) REFERENCES recipes(id)
    )
    ''')
    
    # Create nutrition facts table with column names matching the dataset
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS nutrition (
        food TEXT PRIMARY KEY,
        food_normalized TEXT,
        vitamin_c REAL,         -- Vitamin C
        vitamin_b11 REAL,       -- Vitamin B11
        sodium REAL,            -- Sodium
        calcium REAL,           -- Calcium
        carbohydrates REAL,     -- Carbohydrates
        iron REAL,              -- Iron
        caloric_value REAL,     -- Caloric Value
        sugars REAL,            -- Sugars
        dietary_fiber REAL,     -- Dietary Fiber
        fat REAL,               -- Fat
        protein REAL            -- Protein
    )
    ''')
    
    # Create vector references table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vector_refs (
        id INTEGER PRIMARY KEY,
        table_name TEXT NOT NULL,
        record_id INTEGER NOT NULL,
        vector_id TEXT NOT NULL,
        vector_type TEXT NOT NULL
    )
    ''')
    
    # Create user preferences table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_preferences (
        user_id INTEGER PRIMARY KEY,
        dietary_restrictions TEXT,  -- JSON array
        allergies TEXT,             -- JSON array
        favorite_cuisines TEXT,     -- JSON array
        disliked_ingredients TEXT,  -- JSON array
        health_goals TEXT,          -- JSON object
        techniques TEXT,            -- JSON array of techniques
        items TEXT,                 -- JSON array of items
        n_items INTEGER,            -- Number of items
        ratings TEXT,               -- JSON object of ratings
        n_ratings INTEGER           -- Number of ratings
    )
    ''')
    
    conn.commit()
    print("SQLite database setup complete!")
    return conn

def setup_vector_database(vector_db_path=VECTOR_DB_PATH, use_gpu=True):
    """Set up ChromaDB for vector embeddings storage with optional GPU support."""
    print(f"Setting up ChromaDB for vector storage at: {vector_db_path}")
    
    # Try a more robust approach with an in-memory client
    try:
        # First, ensure the directory exists with proper permissions
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Initialize ChromaDB in-memory client
        print("Initializing ChromaDB in-memory client...")
        chroma_client = chromadb.Client()
        
        # Check if we should use GPU acceleration
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"✅ CUDA is available! Using GPU acceleration with device: {torch.cuda.get_device_name(0)}")
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
                else:
                    print("⚠️ CUDA not available. Falling back to CPU for embeddings.")
            except ImportError:
                print("⚠️ PyTorch not installed. Falling back to CPU for embeddings.")
        
        # Create the embedding function
        try:
            embedding_function = embedding_functions.DefaultEmbeddingFunction()
        except (AttributeError, NameError) as e:
            print(f"Warning: Error creating embedding function: {e}")
            print("Warning: DefaultEmbeddingFunction not available. Using None.")
            embedding_function = None
        
        # Get or create collections with proper error handling
        try:
            # Check if collections already exist (which might happen if the function is called repeatedly)
            existing_collections = [c.name for c in chroma_client.list_collections()]
            print(f"Found existing collections: {existing_collections}")
            
            if "recipe_embeddings" in existing_collections:
                recipe_collection = chroma_client.get_collection(name="recipe_embeddings", embedding_function=embedding_function)
                print("Using existing recipe_embeddings collection")
            else:
                recipe_collection = chroma_client.create_collection(
                    name="recipe_embeddings",
                    embedding_function=embedding_function
                )
                print("Created new recipe_embeddings collection")
                
            if "user_embeddings" in existing_collections:
                user_collection = chroma_client.get_collection(name="user_embeddings", embedding_function=embedding_function)
                print("Using existing user_embeddings collection")
            else:
                user_collection = chroma_client.create_collection(
                    name="user_embeddings",
                    embedding_function=embedding_function
                )
                print("Created new user_embeddings collection")
        
        except Exception as collection_err:
            print(f"Error handling collections: {collection_err}")
            # In case of any issue, recreate the client and try with get_or_create
            chroma_client = chromadb.Client()
            print("Trying to create collections with get_or_create=True...")
            recipe_collection = chroma_client.create_collection(
                name="recipe_embeddings", 
                embedding_function=embedding_function,
                get_or_create=True
            )
            user_collection = chroma_client.create_collection(
                name="user_embeddings", 
                embedding_function=embedding_function,
                get_or_create=True
            )
            
        print("ChromaDB vector database setup complete!")
        return chroma_client, recipe_collection, user_collection
    
    except Exception as e:
        print(f"Error with ChromaDB setup: {e}")
        print("Creating fallback in-memory database with reset collections...")
        
        # Last resort: create new client and delete collections if they exist
        chroma_client = chromadb.Client()
        
        # Safely delete and recreate collections
        try:
            existing_collections = [c.name for c in chroma_client.list_collections()]
            if "recipe_embeddings" in existing_collections:
                chroma_client.delete_collection(name="recipe_embeddings")
            if "user_embeddings" in existing_collections:
                chroma_client.delete_collection(name="user_embeddings")
        except Exception as delete_err:
            print(f"Error deleting collections: {delete_err}")
            
        # Create fresh collections
        recipe_collection = chroma_client.create_collection(name="recipe_embeddings")
        user_collection = chroma_client.create_collection(name="user_embeddings")
        
        return chroma_client, recipe_collection, user_collection

# ============================================================
# PART 1.2: Load DataFrames into Databases
# ============================================================

def load_recipes_to_sqlite(conn: sqlite3.Connection, recipes_df: pd.DataFrame) -> None:
    """Load recipes dataframe into the SQLite database."""
    print(f"Loading {len(recipes_df)} recipes into SQLite...")
    
    # Ensure JSON columns are properly serialized
    recipes_df = recipes_df.copy()
    
    # Fill NULL values in the name column with a default value to satisfy NOT NULL constraint
    if 'name' in recipes_df.columns:
        recipes_df['name'] = recipes_df['name'].fillna('Unnamed Recipe')
        print(f"Filled {recipes_df['name'].isna().sum()} NULL values in name column with 'Unnamed Recipe'")
    
    # Handle JSON serialization for relevant columns
    json_columns = ['tags', 'steps', 'ingredients', 'nutrition', 'normalized_ingredients', 'dietary_tags']
    for col in json_columns:
        if col in recipes_df.columns:
            # Use a safer approach to handle all data types
            def serialize_json(x):
                if isinstance(x, (list, dict)):
                    return json.dumps(x)
                elif isinstance(x, np.ndarray):
                    return json.dumps(x.tolist())
                elif pd.isna(x):
                    return None
                else:
                    try:
                        # Try to convert to JSON - if it fails, convert to string
                        return json.dumps(x)
                    except (TypeError, ValueError):
                        return str(x)
            
            recipes_df[col] = recipes_df[col].apply(serialize_json)
    
    # Get existing table columns
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(recipes)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        
        if existing_columns:
            print(f"Existing columns in recipes table: {existing_columns}")
            # Filter the dataframe to only include columns that exist in the table
            valid_columns = [col for col in recipes_df.columns if col in existing_columns]
            
            if len(valid_columns) < len(recipes_df.columns):
                missing_columns = set(recipes_df.columns) - set(existing_columns)
                print(f"Warning: Dropping columns not in table schema: {missing_columns}")
                
            recipes_df = recipes_df[valid_columns]
    except sqlite3.OperationalError:
        # Table doesn't exist yet, we'll create it
        print("Table 'recipes' doesn't exist yet. It will be created.")
    
    # Insert into database
    recipes_df.to_sql('recipes', conn, if_exists='append', index=False)
    print(f"✅ Successfully loaded {len(recipes_df)} recipes into SQLite")

def load_interactions_to_sqlite(conn: sqlite3.Connection, interactions_df: pd.DataFrame) -> None:
    """Load interactions dataframe into the SQLite database."""
    print(f"Loading {len(interactions_df)} interactions into SQLite...")
    
    # Insert into database
    interactions_df.to_sql('interactions', conn, if_exists='append', index=False)
    print(f"✅ Successfully loaded {len(interactions_df)} interactions into SQLite")

def load_nutrition_to_sqlite(conn: sqlite3.Connection, nutrition_df: pd.DataFrame) -> None:
    """Load nutrition dataframe into the SQLite database."""
    print(f"Loading {len(nutrition_df)} nutrition entries into SQLite...")
    
    # Create a copy of the dataframe to avoid modifying the original
    nutrition_df = nutrition_df.copy()
    
    # Get column mappings from DataFrame to SQLite table schema
    column_mappings = {
        'Vitamin C': 'vitamin_c',
        'Vitamin B11': 'vitamin_b11',
        'Sodium': 'sodium',
        'Calcium': 'calcium',
        'Carbohydrates': 'carbohydrates',
        'Iron': 'iron',
        'Caloric Value': 'caloric_value',
        'Sugars': 'sugars',
        'Dietary Fiber': 'dietary_fiber',
        'Fat': 'fat',
        'Protein': 'protein',
        'food': 'food',  # Keep as is
        'food_normalized': 'food_normalized'  # Keep as is
    }
    
    # Check existing table columns to validate the mapping
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(nutrition)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        print(f"Existing columns in nutrition table: {existing_columns}")
        
        # Rename columns to match SQLite schema
        renamed_columns = {}
        for col in nutrition_df.columns:
            if col in column_mappings and column_mappings[col] in existing_columns:
                renamed_columns[col] = column_mappings[col]
            elif col.lower().replace(' ', '_') in existing_columns:
                # Try automatically converting names (spaces to underscores, lowercase)
                renamed_columns[col] = col.lower().replace(' ', '_')
            else:
                print(f"Warning: Column '{col}' doesn't match any schema column and will be dropped")
        
        # Apply the renaming
        nutrition_df = nutrition_df.rename(columns=renamed_columns)
        
        # Keep only columns that exist in the table
        valid_columns = [col for col in nutrition_df.columns if col in existing_columns]
        if len(valid_columns) < len(nutrition_df.columns):
            missing_columns = set(nutrition_df.columns) - set(existing_columns)
            print(f"Warning: Dropping columns not in table schema: {missing_columns}")
            
        nutrition_df = nutrition_df[valid_columns]
        
    except sqlite3.OperationalError:
        # Table doesn't exist yet, we'll use our predefined mappings
        print("Table 'nutrition' doesn't exist yet. Using predefined column mappings.")
        nutrition_df = nutrition_df.rename(columns=column_mappings)
        # Keep only mapped columns
        nutrition_df = nutrition_df[[col for col in nutrition_df.columns if col in column_mappings.values()]]
    
    # Insert into database
    nutrition_df.to_sql('nutrition', conn, if_exists='append', index=False)
    print(f"✅ Successfully loaded {len(nutrition_df)} nutrition entries into SQLite")

def _process_recipe_batch(batch, expected_columns):
    """Process a batch of recipes for parallel execution."""
    ids = []
    metadatas = []
    documents = []
    
    for _, row in batch.iterrows():
        recipe_id = int(row['id'])
        
        # For this preprocessed data, we'll just store the tokenized data as documents
        # and rely on the search functionality of ChromaDB
        document_text = {}
        
        # Add name tokens if available
        if 'name_tokens' in row and isinstance(row['name_tokens'], list):
            document_text['name_tokens'] = str(row['name_tokens'])
        
        # Add ingredient tokens if available
        if 'ingredient_tokens' in row and isinstance(row['ingredient_tokens'], list):
            document_text['ingredient_tokens'] = str(row['ingredient_tokens'])
            
        # Add steps tokens if available
        if 'steps_tokens' in row and isinstance(row['steps_tokens'], list):
            document_text['steps_tokens'] = str(row['steps_tokens'])
            
        # Combine all available info
        combined_text = str(document_text)
        
        ids.append(f"recipe_{recipe_id}")
        metadatas.append({
            "recipe_id": recipe_id,
            "i": int(row.get('i', 0)),
            "calorie_level": int(row.get('calorie_level', 0)),
            "type": "recipe"
        })
        documents.append(combined_text)
    
    return ids, metadatas, documents

def load_recipes_to_vector_db(recipe_collection, vectorized_recipes_df: pd.DataFrame) -> None:
    """Load preprocessed recipe data into ChromaDB using tokenized data.
    This version handles pre-processed data from Food.com dataset where recipes are already tokenized."""
    print(f"Loading {len(vectorized_recipes_df)} preprocessed recipes into ChromaDB...")
    
    # Check if the dataframe has the expected columns
    expected_columns = ['id', 'name_tokens', 'ingredient_tokens', 'steps_tokens']
    missing_columns = [col for col in expected_columns if col not in vectorized_recipes_df.columns]
    
    if missing_columns:
        print(f"Warning: Missing expected columns in vectorized_recipes_df: {missing_columns}")
        print(f"Available columns: {vectorized_recipes_df.columns.tolist()}")
    
    # Set environment variable to avoid tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Check for existing IDs to avoid duplicates
    try:
        # Get count of existing items
        existing_count = recipe_collection.count()
        if existing_count > 0:
            print(f"Found {existing_count} existing recipes in ChromaDB collection")
            print("Clearing existing recipes to avoid duplicates...")
            # Get all existing IDs
            existing_ids = recipe_collection.get()['ids']
            # Delete them in batches to avoid memory issues
            batch_size = 5000
            for i in range(0, len(existing_ids), batch_size):
                recipe_collection.delete(ids=existing_ids[i:i+batch_size])
            print("Existing recipes cleared successfully")
    except Exception as e:
        print(f"Warning when checking existing recipes: {e}")
    
    # Determine batch size based on available memory
    total_rows = len(vectorized_recipes_df)
    batch_size = 2000  # A reasonable batch size to avoid memory issues
    
    # Split dataframe into batches
    batches = [vectorized_recipes_df.iloc[i:i+batch_size] for i in range(0, total_rows, batch_size)]
    print(f"Processing {len(batches)} batches with approximately {batch_size} recipes per batch")
    
    # Initialize counter for added records
    count = 0
    # Track duplicates to suppress repetitive warnings
    duplicates_warning_count = 0
    max_warnings = 5  # Only show first few warnings
    
    # Process batches sequentially to avoid database write errors
    with tqdm(total=len(batches), desc="Processing recipe batches") as pbar:
        for batch in batches:
            try:
                # Process data for this batch
                ids = []
                metadatas = []
                documents = []
                
                for _, row in batch.iterrows():
                    recipe_id = int(row['id'])
                    
                    # For this preprocessed data, we'll just store the tokenized data as documents
                    # and rely on the search functionality of ChromaDB
                    document_text = {}
                    
                    # Add name tokens if available
                    if 'name_tokens' in row and isinstance(row['name_tokens'], list):
                        document_text['name_tokens'] = str(row['name_tokens'])
                    
                    # Add ingredient tokens if available
                    if 'ingredient_tokens' in row and isinstance(row['ingredient_tokens'], list):
                        document_text['ingredient_tokens'] = str(row['ingredient_tokens'])
                        
                    # Add steps tokens if available
                    if 'steps_tokens' in row and isinstance(row['steps_tokens'], list):
                        document_text['steps_tokens'] = str(row['steps_tokens'])
                        
                    # Combine all available info
                    combined_text = str(document_text)
                    
                    ids.append(f"recipe_{recipe_id}")
                    metadatas.append({
                        "recipe_id": recipe_id,
                        "i": int(row.get('i', 0)),
                        "calorie_level": int(row.get('calorie_level', 0)),
                        "type": "recipe"
                    })
                    documents.append(combined_text)
                
                # Add to ChromaDB in a single batch, with duplicate handling
                if ids:
                    try:
                        recipe_collection.add(
                            ids=ids,
                            metadatas=metadatas,
                            documents=documents
                        )
                        count += len(ids)
                    except Exception as batch_error:
                        if "already exists" in str(batch_error).lower():
                            # Handle duplicate IDs by adding them one by one
                            duplicates_warning_count += 1
                            if duplicates_warning_count <= max_warnings:
                                print(f"Warning: Duplicate IDs found, adding one by one (warning {duplicates_warning_count}/{max_warnings})")
                            
                            # Try adding one by one to skip duplicates
                            for i in range(len(ids)):
                                try:
                                    recipe_collection.add(
                                        ids=[ids[i]],
                                        metadatas=[metadatas[i]],
                                        documents=[documents[i]]
                                    )
                                    count += 1
                                except Exception:
                                    # Skip this item if it's a duplicate
                                    pass
                        else:
                            # For other errors, print the message
                            print(f"Error adding batch: {batch_error}")
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
            
            pbar.update(1)
    
    print(f"✅ Successfully added {count} preprocessed recipes to ChromaDB")

def _process_user_batch(batch, user_id_column):
    """Process a batch of users for parallel execution."""
    ids = []
    metadatas = []
    documents = []
    
    for _, row in batch.iterrows():
        user_id = int(row[user_id_column])
        
        # Combine user information into a document
        user_data = {}
        
        # Add techniques if available
        if 'techniques' in row and row['techniques'] is not None:
            user_data['techniques'] = str(row['techniques'])
            
        # Add items if available
        if 'items' in row and row['items'] is not None:
            user_data['items'] = str(row['items'])
            
        # Add ratings if available
        if 'ratings' in row and row['ratings'] is not None:
            user_data['ratings'] = str(row['ratings'])
        
        # Combine all user data
        combined_text = str(user_data)
        
        ids.append(f"user_{user_id}")
        metadatas.append({
            "user_id": user_id,
            "n_items": int(row.get('n_items', 0)),
            "n_ratings": int(row.get('n_ratings', 0)),
            "type": "user"
        })
        documents.append(combined_text)
    
    return ids, metadatas, documents

def load_users_to_vector_db(user_collection, vectorized_users_df: pd.DataFrame) -> None:
    """Load preprocessed user data into ChromaDB.
    This version handles pre-processed user data from Food.com dataset with techniques, items, and ratings."""
    print(f"Loading {len(vectorized_users_df)} preprocessed users into ChromaDB...")
    
    # Check column names and map to expected names
    user_id_column = 'user_id' if 'user_id' in vectorized_users_df.columns else ('u' if 'u' in vectorized_users_df.columns else None)
    
    if user_id_column is None:
        print("Error: Could not find user ID column in the dataframe. Expected 'user_id' or 'u'.")
        print("Available columns:", vectorized_users_df.columns.tolist())
        return
        
    print(f"Using '{user_id_column}' as the user ID column")
    
    # Set environment variable to avoid tokenizer warnings (if not already set)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Determine batch size based on available memory
    total_rows = len(vectorized_users_df)
    batch_size = 2000  # A reasonable batch size to avoid memory issues
    
    # Split dataframe into batches
    batches = [vectorized_users_df.iloc[i:i+batch_size] for i in range(0, total_rows, batch_size)]
    print(f"Processing {len(batches)} batches with approximately {batch_size} users per batch")
    
    # Initialize counter for added records
    count = 0
    
    # Process batches sequentially to avoid database write errors
    with tqdm(total=len(batches), desc="Processing user batches") as pbar:
        for batch in batches:
            try:
                # Process data for this batch
                ids = []
                metadatas = []
                documents = []
                
                for _, row in batch.iterrows():
                    user_id = int(row[user_id_column])
                    
                    # Combine user information into a document
                    user_data = {}
                    
                    # Add techniques if available
                    if 'techniques' in row and row['techniques'] is not None:
                        user_data['techniques'] = str(row['techniques'])
                        
                    # Add items if available
                    if 'items' in row and row['items'] is not None:
                        user_data['items'] = str(row['items'])
                        
                    # Add ratings if available
                    if 'ratings' in row and row['ratings'] is not None:
                        user_data['ratings'] = str(row['ratings'])
                    
                    # Combine all user data
                    combined_text = str(user_data)
                    
                    ids.append(f"user_{user_id}")
                    metadatas.append({
                        "user_id": user_id,
                        "n_items": int(row.get('n_items', 0)),
                        "n_ratings": int(row.get('n_ratings', 0)),
                        "type": "user"
                    })
                    documents.append(combined_text)
                
                # Add to ChromaDB in a single batch
                if ids:
                    user_collection.add(
                        ids=ids,
                        metadatas=metadatas,
                        documents=documents
                    )
                    count += len(ids)
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
            
            pbar.update(1)
    
    print(f"✅ Successfully added {count} preprocessed users to ChromaDB")

def track_vector_references(conn: sqlite3.Connection, table_name: str, record_id: int, vector_id: str, vector_type: str) -> None:
    """Add a reference to a vector in the vector_refs table."""
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO vector_refs (table_name, record_id, vector_id, vector_type)
    VALUES (?, ?, ?, ?)
    ''', (table_name, record_id, vector_id, vector_type))
    conn.commit()

def track_vector_references_batch(conn: sqlite3.Connection, references: List[Tuple[str, int, str, str]]) -> None:
    """Add multiple references to vectors in the vector_refs table in a single transaction."""
    cursor = conn.cursor()
    cursor.executemany('''
    INSERT INTO vector_refs (table_name, record_id, vector_id, vector_type)
    VALUES (?, ?, ?, ?)
    ''', references)
    conn.commit()

# ============================================================
# PART 1.3: Initialize Databases and Load Data
# ============================================================

def initialize_databases(db_path=DB_PATH):
    """Initialize both SQLite and ChromaDB databases."""
    print("Initializing empty databases...")
    print(f"Using SQLite database at: {db_path}")

    # Initialize SQLite database with the provided path
    sqlite_conn = setup_sqlite_database(db_path)

    # Initialize ChromaDB
    chroma_client, recipe_collection, user_collection = setup_vector_database()
    
    print("Empty databases initialized successfully!")
    return sqlite_conn, chroma_client, recipe_collection, user_collection

def load_all_data_to_databases(
    sqlite_conn: sqlite3.Connection,
    chroma_client,
    recipe_collection,
    user_collection,
    recipes_df: pd.DataFrame = None,
    interactions_df: pd.DataFrame = None,
    nutrition_df: pd.DataFrame = None,
    vectorized_recipes_df: pd.DataFrame = None,
    vectorized_users_df: pd.DataFrame = None
):
    """Load all dataframes directly into the databases."""
    print("Loading data into databases...")
    
    # Load recipe data into SQLite if provided
    if recipes_df is not None:
        load_recipes_to_sqlite(sqlite_conn, recipes_df)
        
    # Load interactions data into SQLite if provided
    if interactions_df is not None:
        load_interactions_to_sqlite(sqlite_conn, interactions_df)
        
    # Load nutrition data into SQLite if provided
    if nutrition_df is not None:
        load_nutrition_to_sqlite(sqlite_conn, nutrition_df)
        
    # Load vectorized recipe data into ChromaDB if provided
    if vectorized_recipes_df is not None:
        load_recipes_to_vector_db(recipe_collection, vectorized_recipes_df)
        
        # Add vector references to SQLite for cross-referencing in batches
        print("Adding vector references for recipes to SQLite...")
        batch_size = 1000
        references = []
        
        for _, row in tqdm(vectorized_recipes_df.iterrows(), total=len(vectorized_recipes_df), desc="Creating recipe vector references"):
            recipe_id = row['id']
            references.append(('recipes', recipe_id, f"recipe_{recipe_id}", 'preprocessed'))
            
            # Process in batches to avoid memory issues
            if len(references) >= batch_size:
                track_vector_references_batch(sqlite_conn, references)
                references = []
        
        # Process any remaining references
        if references:
            track_vector_references_batch(sqlite_conn, references)
    
    # Load vectorized user data into ChromaDB if provided
    if vectorized_users_df is not None:
        load_users_to_vector_db(user_collection, vectorized_users_df)
        
        # Add vector references to SQLite for cross-referencing in batches
        user_id_column = 'user_id' if 'user_id' in vectorized_users_df.columns else ('u' if 'u' in vectorized_users_df.columns else None)
        if user_id_column:
            print("Adding vector references for users to SQLite...")
            batch_size = 1000
            references = []
            
            for _, row in tqdm(vectorized_users_df.iterrows(), total=len(vectorized_users_df), desc="Creating user vector references"):
                user_id = row[user_id_column]
                references.append(('user_preferences', user_id, f"user_{user_id}", 'preprocessed'))
                
                # Process in batches to avoid memory issues
                if len(references) >= batch_size:
                    track_vector_references_batch(sqlite_conn, references)
                    references = []
            
            # Process any remaining references
            if references:
                track_vector_references_batch(sqlite_conn, references)
    
    print("All data successfully loaded into databases!")
    return True

# ============================================================
# PART 1.4: Data Loading Utilities
# ============================================================

def load_data_from_files(
    recipes_file: str = None,
    interactions_file: str = None,
    nutrition_file: str = None,
    vectorized_recipes_file: str = None,
    vectorized_users_file: str = None
) -> Dict[str, pd.DataFrame]:
    """Load data from files into pandas DataFrames."""
    print("Loading data from files...")
    dataframes = {}
    
    # Load recipes data if file exists
    if recipes_file and os.path.exists(recipes_file):
        if recipes_file.endswith('.csv'):
            dataframes['recipes_df'] = pd.read_csv(recipes_file)
        elif recipes_file.endswith('.json'):
            dataframes['recipes_df'] = pd.read_json(recipes_file)
        print(f"✅ Loaded {len(dataframes['recipes_df'])} recipes from {recipes_file}")
    
    # Load interactions data if file exists
    if interactions_file and os.path.exists(interactions_file):
        if interactions_file.endswith('.csv'):
            dataframes['interactions_df'] = pd.read_csv(interactions_file)
        elif interactions_file.endswith('.json'):
            dataframes['interactions_df'] = pd.read_json(interactions_file)
        print(f"✅ Loaded {len(dataframes['interactions_df'])} interactions from {interactions_file}")
    
    # Load nutrition data if file exists
    if nutrition_file and os.path.exists(nutrition_file):
        if nutrition_file.endswith('.csv'):
            dataframes['nutrition_df'] = pd.read_csv(nutrition_file)
        elif nutrition_file.endswith('.json'):
            dataframes['nutrition_df'] = pd.read_json(nutrition_file)
        print(f"✅ Loaded {len(dataframes['nutrition_df'])} nutrition entries from {nutrition_file}")
    
    # Load vectorized recipes data if file exists
    if vectorized_recipes_file and os.path.exists(vectorized_recipes_file):
        if vectorized_recipes_file.endswith('.csv'):
            dataframes['vectorized_recipes_df'] = pd.read_csv(vectorized_recipes_file)
        elif vectorized_recipes_file.endswith('.json'):
            dataframes['vectorized_recipes_df'] = pd.read_json(vectorized_recipes_file)
            
            # Convert string embeddings back to numpy arrays if needed
            for col in dataframes['vectorized_recipes_df'].columns:
                if '_embedding' in col:
                    dataframes['vectorized_recipes_df'][col] = dataframes['vectorized_recipes_df'][col].apply(
                        lambda x: np.array(x) if isinstance(x, list) else x
                    )
        print(f"✅ Loaded {len(dataframes['vectorized_recipes_df'])} vectorized recipes from {vectorized_recipes_file}")
    
    # Load vectorized users data if file exists
    if vectorized_users_file and os.path.exists(vectorized_users_file):
        if vectorized_users_file.endswith('.csv'):
            dataframes['vectorized_users_df'] = pd.read_csv(vectorized_users_file)
        elif vectorized_users_file.endswith('.json'):
            dataframes['vectorized_users_df'] = pd.read_json(vectorized_users_file)
            
            # Convert string embeddings back to numpy arrays if needed
            for col in dataframes['vectorized_users_df'].columns:
                if '_embedding' in col:
                    dataframes['vectorized_users_df'][col] = dataframes['vectorized_users_df'][col].apply(
                        lambda x: np.array(x) if isinstance(x, list) else x
                    )
        print(f"✅ Loaded {len(dataframes['vectorized_users_df'])} vectorized users from {vectorized_users_file}")
    
    return dataframes

def setup_and_load_databases_from_dataframes(
    db_path: str = DB_PATH,
    recipes_df: pd.DataFrame = None,
    interactions_df: pd.DataFrame = None,
    nutrition_df: pd.DataFrame = None,
    vectorized_recipes_df: pd.DataFrame = None,
    vectorized_users_df: pd.DataFrame = None
) -> Tuple[sqlite3.Connection, Any, Any, Any]:
    """
    One-step function to set up databases and load data from dataframes.
    
    Args:
        db_path: Path to SQLite database file
        recipes_df: DataFrame containing recipe data
        interactions_df: DataFrame containing user-recipe interactions
        nutrition_df: DataFrame containing nutrition data
        vectorized_recipes_df: DataFrame containing recipe embeddings
        vectorized_users_df: DataFrame containing user embeddings
    
    Returns:
        Tuple containing SQLite connection, ChromaDB client, recipe collection, and user collection
    """
    # Initialize empty databases
    sqlite_conn, chroma_client, recipe_collection, user_collection = initialize_databases(db_path)
    
    # Load DataFrames into databases
    load_all_data_to_databases(
        sqlite_conn,
        chroma_client,
        recipe_collection,
        user_collection,
        recipes_df=recipes_df,
        interactions_df=interactions_df,
        nutrition_df=nutrition_df,
        vectorized_recipes_df=vectorized_recipes_df,
        vectorized_users_df=vectorized_users_df
    )
    
    return sqlite_conn, chroma_client, recipe_collection, user_collection

# We'll keep the original function for loading from files as a backup option
def setup_and_load_databases_from_files(
    db_path: str = DB_PATH,
    recipes_file: str = None,
    interactions_file: str = None,
    nutrition_file: str = None,
    vectorized_recipes_file: str = None,
    vectorized_users_file: str = None
) -> Tuple[sqlite3.Connection, Any, Any, Any]:
    """
    One-step function to set up databases and load data from files.
    
    Returns:
        Tuple containing SQLite connection, ChromaDB client, recipe collection, and user collection
    """
    # Initialize empty databases
    sqlite_conn, chroma_client, recipe_collection, user_collection = initialize_databases(db_path)
    
    # Load data from files into DataFrames
    dataframes = load_data_from_files(
        recipes_file,
        interactions_file,
        nutrition_file,
        vectorized_recipes_file,
        vectorized_users_file
    )
    
    # Load DataFrames into databases
    load_all_data_to_databases(
        sqlite_conn,
        chroma_client,
        recipe_collection,
        user_collection,
        recipes_df=dataframes.get('recipes_df'),
        interactions_df=dataframes.get('interactions_df'),
        nutrition_df=dataframes.get('nutrition_df'),
        vectorized_recipes_df=dataframes.get('vectorized_recipes_df'),
        vectorized_users_df=dataframes.get('vectorized_users_df')
    )
    
    return sqlite_conn, chroma_client, recipe_collection, user_collection

def export_chromadb_to_disk(chroma_client, vector_db_path=VECTOR_DB_PATH):
    """Export ChromaDB in-memory collections to disk.
    This is a workaround for the file permission issues with the PersistentClient."""
    try:
        print(f"Exporting ChromaDB collections to disk at {vector_db_path}...")
        
        # Make sure the directory exists with proper permissions
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Get collection names - in v0.6.0, list_collections returns just the names
        collection_names = chroma_client.list_collections()
        print(f"Found {len(collection_names)} collections to export: {collection_names}")
        
        for collection_name in collection_names:
            try:
                # Get the collection - API change in v0.6.0
                collection = chroma_client.get_collection(name=collection_name)
                
                # Get all data from the collection
                all_data = collection.get(include=['embeddings', 'documents', 'metadatas'])
                
                if not all_data['ids'] or len(all_data['ids']) == 0:
                    print(f"Collection {collection_name} is empty, skipping export")
                    continue
                    
                # Save the collection data to a pickle file
                collection_path = os.path.join(vector_db_path, f"{collection_name}.pkl")
                with open(collection_path, 'wb') as f:
                    import pickle
                    pickle.dump(all_data, f)
                
                print(f"✅ Successfully exported collection {collection_name} with {len(all_data['ids'])} items")
            except Exception as inner_error:
                print(f"Error exporting collection {collection_name}: {inner_error}")
                # Continue with other collections even if one fails
                continue
            
        print("Collections export process completed!")
        return True
    except Exception as e:
        print(f"Error exporting ChromaDB to disk: {e}")
        return False

# Main function using dataframes directly
if __name__ == "__main__":
    # Load data directly to dataframes
    import pandas as pd
    import os
    
    # Define base directory and absolute file paths
    base_dir = os.path.expanduser("~/coding/python/google_capstone")
    recipes_file = os.path.join(base_dir, "datasets/RAW_recipes.csv")
    interactions_file = os.path.join(base_dir, "datasets/RAW_interactions.csv")
    nutrition_file = os.path.join(base_dir, "datasets/cleaned_nutrition_dataset.csv")
    vectorized_recipes_file = os.path.join(base_dir, "datasets/PP_recipes.csv")
    vectorized_users_file = os.path.join(base_dir, "datasets/PP_users.csv")
    
    print(f"Using dataset directory: {os.path.join(base_dir, 'datasets')}")
    print("Loading data into pandas dataframes...")
    
    try:
        recipes_df = pd.read_csv(recipes_file)
        print(f"✅ Loaded {len(recipes_df)} recipes")
    except Exception as e:
        print(f"Warning: Could not load recipes file: {e}")
        recipes_df = None
    
    try:
        interactions_df = pd.read_csv(interactions_file)
        print(f"✅ Loaded {len(interactions_df)} interactions")
    except Exception as e:
        print(f"Warning: Could not load interactions file: {e}")
        interactions_df = None
        
    try:
        nutrition_df = pd.read_csv(nutrition_file)
        print(f"✅ Loaded {len(nutrition_df)} nutrition entries")
    except Exception as e:
        print(f"Warning: Could not load nutrition file: {e}")
        nutrition_df = None
    
    try:
        vectorized_recipes_df = pd.read_csv(vectorized_recipes_file)
        print(f"✅ Loaded {len(vectorized_recipes_df)} vectorized recipes")
    except Exception as e:
        print(f"Warning: Could not load vectorized recipes file: {e}")
        vectorized_recipes_df = None
    
    try:
        vectorized_users_df = pd.read_csv(vectorized_users_file)
        print(f"✅ Loaded {len(vectorized_users_df)} vectorized users")
    except Exception as e:
        print(f"Warning: Could not load vectorized users file: {e}")
        vectorized_users_df = None
    
    # Initialize databases
    print("\nInitializing databases...")
    sqlite_conn, chroma_client, recipe_collection, user_collection = initialize_databases()
    
    # Load the data into databases
    load_all_data_to_databases(
        sqlite_conn,
        chroma_client,
        recipe_collection,
        user_collection,
        recipes_df=recipes_df,
        interactions_df=interactions_df,
        nutrition_df=nutrition_df,
        vectorized_recipes_df=vectorized_recipes_df,
        vectorized_users_df=vectorized_users_df
    )
    
    # Export ChromaDB collections to disk
    success = export_chromadb_to_disk(chroma_client)
    if success:
        print("\nVector database successfully exported to disk!")
    
    print("\nDatabase setup complete! You can now use the databases for your application.")
    print("- SQLite database is available at:", DB_PATH)
    print("- Vector database is available at:", VECTOR_DB_PATH)
