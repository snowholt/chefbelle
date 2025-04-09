# Step 3: Recipe Customization with Function Calling and Hybrid Storage
# Interactive Recipe & Kitchen Management Assistant

import os
import json
import numpy as np
import pandas as pd
import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Tuple, Optional
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import seaborn as sns
from google import genai

# Set up Google API credentials from environment variable
# In a real implementation, use a more secure approach for API keys
API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=API_KEY)

print("# Step 3: Recipe Customization with Function Calling and Hybrid Storage")
print("## Setting up the hybrid database architecture")

# Define paths and connection variables
DB_PATH = "recipes_db.sqlite"
SAMPLE_DATA_PATH = "sample_recipes.json"  # For demonstration purposes

# ============================================================
# PART 1: Hybrid Database Setup
# ============================================================

def setup_sqlite_database(db_path: str = DB_PATH):
    """Set up SQLite database with the required schema."""
    print("Setting up SQLite database...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create recipes table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS recipes (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        minutes INTEGER,
        submitted TEXT,
        description TEXT,
        n_steps INTEGER,
        steps TEXT,           -- JSON array of steps
        n_ingredients INTEGER,
        ingredients TEXT,     -- JSON array of ingredients
        cuisine_type TEXT,
        nutrition TEXT,       -- JSON object with nutrition info
        rating REAL,
        n_ratings INTEGER
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
    
    # Create nutrition facts table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS nutrition (
        food TEXT PRIMARY KEY,
        food_normalized TEXT,
        calories REAL,
        fat REAL,
        protein REAL,
        carbs REAL,
        fiber REAL,
        sugar REAL,
        sodium REAL,
        calcium REAL,
        iron REAL,
        vitamin_c REAL,
        vitamin_a REAL
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
        health_goals TEXT           -- JSON object
    )
    ''')
    
    conn.commit()
    print("SQLite database setup complete!")
    return conn

def setup_vector_database():
    """Set up ChromaDB for vector embeddings storage."""
    print("Setting up ChromaDB for vector storage...")
    
    # Initialize ChromaDB client
    chroma_client = chromadb.Client()
    
    # Set up embeddings function - use the all-MiniLM model
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    
    # Create collections
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

def load_sample_data(db_conn, recipe_collection):
    """Load sample data for demonstration purposes."""
    print("Loading sample recipe data...")
    
    # For demonstration, we'll create a small set of sample recipes
    sample_recipes = [
        {
            "id": 1,
            "name": "Creamy Garlic Pasta",
            "minutes": 30,
            "submitted": "2022-01-15",
            "description": "A rich and creamy pasta dish with garlic and parmesan.",
            "n_steps": 5,
            "steps": json.dumps([
                "Boil pasta according to package directions.",
                "In a pan, sauté minced garlic in butter until fragrant.",
                "Add heavy cream and bring to a simmer.",
                "Stir in grated parmesan cheese until smooth.",
                "Toss with cooked pasta and garnish with parsley."
            ]),
            "n_ingredients": 7,
            "ingredients": json.dumps([
                "pasta", "butter", "garlic", "heavy cream", 
                "parmesan cheese", "salt", "parsley"
            ]),
            "cuisine_type": "italian",
            "nutrition": json.dumps({
                "calories": 450, "fat": 28, "protein": 12, 
                "carbs": 42, "fiber": 2, "sugar": 3
            }),
            "rating": 4.7,
            "n_ratings": 235
        },
        {
            "id": 2,
            "name": "Vegetable Stir Fry",
            "minutes": 20,
            "submitted": "2022-02-20",
            "description": "A quick and healthy vegetable stir fry with soy sauce.",
            "n_steps": 4,
            "steps": json.dumps([
                "Heat oil in a wok or large pan over high heat.",
                "Add vegetables and stir fry for 3-4 minutes until crisp-tender.",
                "Add soy sauce, ginger, and garlic, stir to combine.",
                "Serve over rice or noodles if desired."
            ]),
            "n_ingredients": 8,
            "ingredients": json.dumps([
                "broccoli", "bell peppers", "carrots", "snap peas", 
                "garlic", "ginger", "soy sauce", "vegetable oil"
            ]),
            "cuisine_type": "asian",
            "nutrition": json.dumps({
                "calories": 180, "fat": 7, "protein": 5, 
                "carbs": 25, "fiber": 6, "sugar": 8
            }),
            "rating": 4.5,
            "n_ratings": 187
        },
        {
            "id": 3,
            "name": "Classic Beef Lasagna",
            "minutes": 90,
            "submitted": "2022-03-05",
            "description": "A hearty beef lasagna with layers of pasta, meat sauce, and cheese.",
            "n_steps": 7,
            "steps": json.dumps([
                "Brown ground beef with onions and garlic.",
                "Add tomato sauce and seasonings, simmer for 15 minutes.",
                "In a bowl, mix ricotta cheese with egg and herbs.",
                "Layer lasagna noodles, meat sauce, ricotta mixture, and mozzarella in a baking dish.",
                "Repeat layers, ending with cheese on top.",
                "Cover with foil and bake at 375°F for 25 minutes.",
                "Remove foil and bake for another 10 minutes until cheese is golden."
            ]),
            "n_ingredients": 12,
            "ingredients": json.dumps([
                "ground beef", "onion", "garlic", "tomato sauce", 
                "lasagna noodles", "ricotta cheese", "mozzarella cheese", 
                "parmesan cheese", "egg", "basil", "oregano", "salt"
            ]),
            "cuisine_type": "italian",
            "nutrition": json.dumps({
                "calories": 520, "fat": 27, "protein": 32, 
                "carbs": 35, "fiber": 3, "sugar": 7
            }),
            "rating": 4.8,
            "n_ratings": 312
        },
        {
            "id": 4,
            "name": "Vegan Buddha Bowl",
            "minutes": 25,
            "submitted": "2022-04-15",
            "description": "A nutritious vegan bowl with grains, vegetables, and tahini dressing.",
            "n_steps": 5,
            "steps": json.dumps([
                "Cook quinoa according to package directions.",
                "Roast sweet potatoes, chickpeas, and kale in the oven until crispy.",
                "Prepare tahini dressing by mixing tahini, lemon juice, and water.",
                "Assemble bowl with quinoa, roasted vegetables, and avocado.",
                "Drizzle with tahini dressing and sprinkle with sesame seeds."
            ]),
            "n_ingredients": 9,
            "ingredients": json.dumps([
                "quinoa", "sweet potatoes", "chickpeas", "kale", 
                "avocado", "tahini", "lemon juice", "sesame seeds", "olive oil"
            ]),
            "cuisine_type": "mediterranean",
            "nutrition": json.dumps({
                "calories": 380, "fat": 18, "protein": 12, 
                "carbs": 48, "fiber": 12, "sugar": 6
            }),
            "rating": 4.6,
            "n_ratings": 158
        }
    ]
    
    cursor = db_conn.cursor()
    
    # Insert sample recipes into the database
    for recipe in sample_recipes:
        cursor.execute('''
        INSERT OR REPLACE INTO recipes 
        (id, name, minutes, submitted, description, n_steps, steps, 
         n_ingredients, ingredients, cuisine_type, nutrition, rating, n_ratings) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            recipe["id"], recipe["name"], recipe["minutes"], 
            recipe["submitted"], recipe["description"], recipe["n_steps"], 
            recipe["steps"], recipe["n_ingredients"], recipe["ingredients"], 
            recipe["cuisine_type"], recipe["nutrition"], recipe["rating"], 
            recipe["n_ratings"]
        ))
    
    db_conn.commit()
    
    # Generate and store embeddings for each recipe
    for recipe in sample_recipes:
        # Create embedding for recipe name
        recipe_collection.add(
            ids=[f"name_{recipe['id']}"],
            documents=[recipe["name"]],
            metadatas=[{"recipe_id": recipe["id"], "type": "name"}]
        )
        
        # Create embedding for recipe ingredients
        ingredients_text = ", ".join(json.loads(recipe["ingredients"]))
        recipe_collection.add(
            ids=[f"ingredients_{recipe['id']}"],
            documents=[ingredients_text],
            metadatas=[{"recipe_id": recipe["id"], "type": "ingredients"}]
        )
        
        # Create embedding for recipe steps
        steps_text = " ".join(json.loads(recipe["steps"]))
        recipe_collection.add(
            ids=[f"steps_{recipe['id']}"],
            documents=[steps_text],
            metadatas=[{"recipe_id": recipe["id"], "type": "steps"}]
        )
    
    print(f"Loaded {len(sample_recipes)} sample recipes with embeddings!")

# ============================================================
# PART 2: Database Query Functions
# ============================================================

def list_tables(conn):
    """List all tables in the SQLite database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]

def describe_table(conn, table_name):
    """Describe the schema of a specified table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema = cursor.fetchall()
    return [(col[1], col[2]) for col in schema]

def execute_query(conn, sql):
    """Execute an SQL query and return the results."""
    cursor = conn.cursor()
    cursor.execute(sql)
    return cursor.fetchall()

def get_recipe_by_id(conn, recipe_id):
    """Get a recipe by its ID with all details."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM recipes WHERE id = ?", (recipe_id,))
    recipe_data = cursor.fetchone()
    
    if not recipe_data:
        return None
    
    columns = [col[0] for col in cursor.description]
    recipe = dict(zip(columns, recipe_data))
    
    # Parse JSON fields
    for field in ["steps", "ingredients", "nutrition"]:
        if recipe[field]:
            recipe[field] = json.loads(recipe[field])
    
    return recipe

def search_recipes_by_text(recipe_collection, query_text, n_results=5):
    """Search recipes using vector similarity on text."""
    results = recipe_collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    # Extract the recipe IDs
    recipe_ids = []
    for metadata in results['metadatas'][0]:
        if 'recipe_id' in metadata:
            recipe_id = metadata['recipe_id']
            if recipe_id not in recipe_ids:
                recipe_ids.append(recipe_id)
    
    return recipe_ids

def search_recipes_by_ingredients(conn, recipe_collection, ingredients, n_results=5):
    """Search for recipes that contain the specified ingredients using embeddings."""
    # Convert ingredients list to a single string for embedding
    ingredients_text = ", ".join(ingredients)
    
    # Query the vector database
    results = recipe_collection.query(
        query_texts=[ingredients_text],
        n_results=n_results*2,  # Get more results than needed to filter
        where={"type": "ingredients"}
    )
    
    # Extract recipe IDs from the results
    recipe_ids = []
    for i, metadata in enumerate(results['metadatas'][0]):
        if 'recipe_id' in metadata:
            recipe_id = metadata['recipe_id']
            if recipe_id not in recipe_ids:
                recipe_ids.append(recipe_id)
    
    # Limit to the requested number of results
    recipe_ids = recipe_ids[:n_results]
    
    # Get full recipe details for each ID
    recipes = []
    for recipe_id in recipe_ids:
        recipe = get_recipe_by_id(conn, recipe_id)
        if recipe:
            recipes.append(recipe)
    
    return recipes

def get_similar_recipes(conn, recipe_collection, recipe_id, similarity_type="ingredients", n_results=3):
    """Find recipes similar to the given recipe based on the specified similarity type."""
    # Get the original recipe
    original_recipe = get_recipe_by_id(conn, recipe_id)
    if not original_recipe:
        return []
    
    # Query for similar recipes based on the similarity type
    query_id = f"{similarity_type}_{recipe_id}"
    
    try:
        results = recipe_collection.query(
            ids=[query_id],
            n_results=n_results + 1,  # +1 because it will include the original
            where={"type": similarity_type}
        )
    except Exception as e:
        print(f"Error querying vector database: {e}")
        
        # Fallback: query by text if ID query fails
        if similarity_type == "ingredients":
            query_text = ", ".join(original_recipe["ingredients"])
        elif similarity_type == "steps":
            query_text = " ".join(original_recipe["steps"])
        else:
            query_text = original_recipe["name"]
            
        results = recipe_collection.query(
            query_texts=[query_text],
            n_results=n_results + 1,
            where={"type": similarity_type}
        )
    
    # Extract recipe IDs excluding the original
    recipe_ids = []
    for metadata in results['metadatas'][0]:
        if 'recipe_id' in metadata:
            result_id = metadata['recipe_id']
            if result_id != recipe_id and result_id not in recipe_ids:
                recipe_ids.append(result_id)
    
    # Get complete recipe information
    similar_recipes = []
    for similar_id in recipe_ids[:n_results]:
        recipe = get_recipe_by_id(conn, similar_id)
        if recipe:
            similar_recipes.append(recipe)
    
    return similar_recipes

# ============================================================
# PART 3: Recipe Customization with Function Calling
# ============================================================

# Define function schemas for the Gemini API
def get_function_definitions():
    """Define the schema for functions that can be called by the Gemini API."""
    return [
        {
            "name": "search_recipes_by_query",
            "description": "Search for recipes based on a natural language query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search recipes"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recipes to return"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_recipe_by_id",
            "description": "Get detailed information about a specific recipe",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_id": {
                        "type": "integer",
                        "description": "ID of the recipe to retrieve"
                    }
                },
                "required": ["recipe_id"]
            }
        },
        {
            "name": "customize_recipe",
            "description": "Customize a recipe based on dietary preferences or ingredient substitutions",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_id": {
                        "type": "integer",
                        "description": "ID of the recipe to customize"
                    },
                    "dietary_restrictions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of dietary restrictions (e.g., vegetarian, vegan, gluten-free)"
                    },
                    "ingredients_to_add": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ingredients to add to the recipe"
                    },
                    "ingredients_to_remove": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ingredients to remove from the recipe"
                    },
                    "serving_size": {
                        "type": "integer",
                        "description": "New serving size for the recipe"
                    }
                },
                "required": ["recipe_id"]
            }
        },
        {
            "name": "get_similar_recipes",
            "description": "Find recipes similar to a given recipe",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_id": {
                        "type": "integer",
                        "description": "ID of the reference recipe"
                    },
                    "similarity_type": {
                        "type": "string",
                        "enum": ["ingredients", "steps", "name"],
                        "description": "Type of similarity to consider"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of similar recipes to return"
                    }
                },
                "required": ["recipe_id"]
            }
        }
    ]

def customize_recipe(conn, recipe_id, dietary_restrictions=None, ingredients_to_add=None, 
                     ingredients_to_remove=None, serving_size=None):
    """Customize a recipe based on specified modifications."""
    # Get the original recipe
    original_recipe = get_recipe_by_id(conn, recipe_id)
    if not original_recipe:
        return {"error": f"Recipe with ID {recipe_id} not found"}
    
    # Create a deep copy of the original recipe for modification
    customized_recipe = original_recipe.copy()
    customized_recipe["original_id"] = recipe_id
    customized_recipe["customized"] = True
    customized_recipe["modifications"] = []
    
    # Apply dietary restrictions
    if dietary_restrictions:
        # For each dietary restriction, modify the ingredients accordingly
        for restriction in dietary_restrictions:
            restriction = restriction.lower()
            
            if restriction == "vegetarian":
                meat_ingredients = ["beef", "chicken", "pork", "lamb", "veal", "bacon", "ham", "sausage", "turkey"]
                ingredients_to_remove = ingredients_to_remove or []
                
                # Find meat ingredients in the recipe
                for ingredient in original_recipe["ingredients"]:
                    if any(meat in ingredient.lower() for meat in meat_ingredients):
                        ingredients_to_remove.append(ingredient)
                
                if ingredients_to_remove:
                    customized_recipe["modifications"].append(f"Made vegetarian by removing {', '.join(ingredients_to_remove)}")
            
            elif restriction == "vegan":
                animal_products = [
                    "meat", "beef", "chicken", "pork", "lamb", "veal", "bacon", "ham", "sausage",
                    "milk", "cheese", "cream", "butter", "yogurt", "egg", "honey"
                ]
                ingredients_to_remove = ingredients_to_remove or []
                
                # Find animal products in the recipe
                for ingredient in original_recipe["ingredients"]:
                    if any(product in ingredient.lower() for product in animal_products):
                        ingredients_to_remove.append(ingredient)
                
                if ingredients_to_remove:
                    customized_recipe["modifications"].append(f"Made vegan by removing {', '.join(ingredients_to_remove)}")
            
            elif restriction == "gluten-free":
                gluten_ingredients = ["wheat", "flour", "pasta", "bread", "crackers", "barley", "rye"]
                ingredients_to_remove = ingredients_to_remove or []
                
                # Find gluten-containing ingredients
                for ingredient in original_recipe["ingredients"]:
                    if any(gluten in ingredient.lower() for gluten in gluten_ingredients):
                        ingredients_to_remove.append(ingredient)
                
                if ingredients_to_remove:
                    customized_recipe["modifications"].append(f"Made gluten-free by removing {', '.join(ingredients_to_remove)}")
    
    # Remove specified ingredients
    if ingredients_to_remove:
        customized_ingredients = []
        for ingredient in original_recipe["ingredients"]:
            if not any(remove_ing.lower() in ingredient.lower() for remove_ing in ingredients_to_remove):
                customized_ingredients.append(ingredient)
        
        customized_recipe["ingredients"] = customized_ingredients
        customized_recipe["n_ingredients"] = len(customized_ingredients)
        
        if not any("removing" in mod for mod in customized_recipe["modifications"]):
            customized_recipe["modifications"].append(f"Removed ingredients: {', '.join(ingredients_to_remove)}")
    
    # Add new ingredients
    if ingredients_to_add:
        for ingredient in ingredients_to_add:
            if ingredient not in customized_recipe["ingredients"]:
                customized_recipe["ingredients"].append(ingredient)
        
        customized_recipe["n_ingredients"] = len(customized_recipe["ingredients"])
        customized_recipe["modifications"].append(f"Added ingredients: {', '.join(ingredients_to_add)}")
    
    # Adjust serving size
    if serving_size and serving_size != original_recipe.get("servings", 4):
        original_servings = original_recipe.get("servings", 4)
        multiplier = serving_size / original_servings
        
        # Note about scaling
        customized_recipe["servings"] = serving_size
        customized_recipe["modifications"].append(f"Adjusted serving size from {original_servings} to {serving_size}")
        
        # In a real implementation, we would scale quantities in the ingredients
        # This is a simplified version that just notes the scaling
        customized_recipe["scaling_note"] = f"All ingredient quantities should be multiplied by {multiplier:.2f}"
    
    # Update the recipe name to indicate customization
    if customized_recipe["modifications"]:
        prefix = []
        if dietary_restrictions:
            prefix.extend(r.capitalize() for r in dietary_restrictions)
        if ingredients_to_add:
            prefix.append("Modified")
        
        if prefix:
            customized_recipe["name"] = f"{' '.join(prefix)} {original_recipe['name']}"
    
    return customized_recipe

def handle_function_call(conn, recipe_collection, function_name, function_args):
    """Execute the function called by the Gemini API."""
    if function_name == "search_recipes_by_query":
        query = function_args.get("query", "")
        limit = function_args.get("limit", 5)
        
        # Get recipe IDs from vector search
        recipe_ids = search_recipes_by_text(recipe_collection, query, limit)
        
        # Get complete recipe information
        recipes = []
        for recipe_id in recipe_ids:
            recipe = get_recipe_by_id(conn, recipe_id)
            if recipe:
                # Simplify the output for clarity
                recipes.append({
                    "id": recipe["id"],
                    "name": recipe["name"],
                    "description": recipe["description"],
                    "ingredients": recipe["ingredients"],
                    "cuisine_type": recipe["cuisine_type"],
                    "minutes": recipe["minutes"]
                })
        
        return recipes
    
    elif function_name == "get_recipe_by_id":
        recipe_id = function_args.get("recipe_id")
        if not recipe_id:
            return {"error": "Recipe ID is required"}
        
        recipe = get_recipe_by_id(conn, recipe_id)
        return recipe if recipe else {"error": f"Recipe with ID {recipe_id} not found"}
    
    elif function_name == "customize_recipe":
        recipe_id = function_args.get("recipe_id")
        if not recipe_id:
            return {"error": "Recipe ID is required"}
        
        return customize_recipe(
            conn,
            recipe_id,
            dietary_restrictions=function_args.get("dietary_restrictions"),
            ingredients_to_add=function_args.get("ingredients_to_add"),
            ingredients_to_remove=function_args.get("ingredients_to_remove"),
            serving_size=function_args.get("serving_size")
        )
    
    elif function_name == "get_similar_recipes":
        recipe_id = function_args.get("recipe_id")
        if not recipe_id:
            return {"error": "Recipe ID is required"}
        
        similarity_type = function_args.get("similarity_type", "ingredients")
        limit = function_args.get("limit", 3)
        
        similar_recipes = get_similar_recipes(
            conn, 
            recipe_collection, 
            recipe_id, 
            similarity_type, 
            limit
        )
        
        # Simplify the output for clarity
        simplified_recipes = []
        for recipe in similar_recipes:
            simplified_recipes.append({
                "id": recipe["id"],
                "name": recipe["name"],
                "description": recipe["description"],
                "ingredients": recipe["ingredients"],
                "cuisine_type": recipe["cuisine_type"],
                "minutes": recipe["minutes"]
            })
        
        return simplified_recipes
    
    else:
        return {"error": f"Unknown function: {function_name}"}

# ============================================================
# PART 4: Gemini API Integration for Recipe Customization
# ============================================================

def recipe_customization_chat(conn, recipe_collection, user_query):
    """Implement a chat interface for recipe customization using Gemini API function calling."""
    # Set up the Gemini model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config={"temperature": 0.2}
    )
    
    # Define system prompt
    system_prompt = """
    You are RecipeGenie, an expert AI assistant specialized in recipe customization. 
    Your goal is to help users find, understand, and customize recipes according to their preferences.
    
    You can:
    1. Search for recipes based on ingredients, cuisine types, dietary restrictions, or dish names
    2. Get detailed information about specific recipes
    3. Customize recipes according to dietary needs (vegetarian, vegan, gluten-free, etc.)
    4. Find similar recipes to ones the user is interested in
    
    When customizing recipes:
    - For vegetarian: Replace meat with plant-based proteins like tofu, tempeh, or legumes
    - For vegan: Replace all animal products (meat, dairy, eggs, honey) with plant-based alternatives
    - For gluten-free: Replace wheat flour with alternative flours, pasta with GF pasta, etc.
    
    Respond in a helpful, conversational manner. Always suggest alternatives when removing ingredients.
    """
    
    # Create a chat session
    chat = model.start_chat(tools=get_function_definitions())
    
    # Send system instructions
    chat.send_message(system_prompt)
    
    # Send user query and handle function calls
    response = chat.send_message(user_query)
    
    # Check if the response includes function calls
    if hasattr(response, 'candidates') and response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        function_name = function_call.name
        function_args = json.loads(function_call.args)
        
        print(f"Function called: {function_name}")
        print(f"Arguments: {json.dumps(function_args, indent=2)}")
        
        # Execute the function
        function_response = handle_function_call(conn, recipe_collection, function_name, function_args)
        
        # Send the function response back to the model
        follow_up = chat.send_message(function_response)
        
        # Return both the function data and the model's interpretation
        return {
            "function_called": function_name,
            "function_args": function_args,
            "function_response": function_response,
            "assistant_response": follow_up.text
        }
    else:
        # Return just the text response if no function was called
        return {
            "function_called": None,
            "assistant_response": response.text
        }

# ============================================================
# PART 5: Few-Shot Prompting for Recipe Customization
# ============================================================

def customize_recipe_with_few_shot(conn, recipe_id, user_request):
    """Customize a recipe using few-shot prompting instead of function calling."""
    # Get the original recipe
    original_recipe = get_recipe_by_id(conn, recipe_id)
    if not original_recipe:
        return {"error": f"Recipe with ID {recipe_id} not found"}
    
    # Define few-shot examples
    few_shot_examples = """
    Example 1:
    Original Recipe: Chicken Alfredo Pasta
    Ingredients: chicken breast, fettuccine pasta, heavy cream, butter, parmesan cheese, garlic, salt, pepper
    User Request: "I'm vegetarian, can you adapt this recipe for me?"
    Customized Recipe:
    Vegetarian Alfredo Pasta
    Ingredients: fettuccine pasta, heavy cream, butter, parmesan cheese, garlic, salt, pepper, mushrooms, spinach
    Modifications:
    - Removed chicken breast
    - Added mushrooms and spinach for texture and nutrition
    - Use vegetable broth instead of chicken broth if needed
    
    Example 2:
    Original Recipe: Classic Beef Lasagna
    Ingredients: ground beef, lasagna noodles, ricotta cheese, mozzarella cheese, parmesan cheese, egg, tomato sauce, onion, garlic, herbs
    User Request: "I need a dairy-free version of this recipe"
    Customized Recipe:
    Dairy-Free Beef Lasagna
    Ingredients: ground beef, lasagna noodles, dairy-free ricotta alternative, dairy-free mozzarella alternative, nutritional yeast, tomato sauce, onion, garlic, herbs
    Modifications:
    - Replaced ricotta cheese with silken tofu blended with nutritional yeast and herbs
    - Replaced mozzarella with dairy-free cheese alternative
    - Replaced parmesan with nutritional yeast
    - Removed egg (can use 1 tbsp cornstarch mixed with 2 tbsp water as a binder)
    
    Example 3:
    Original Recipe: Chocolate Chip Cookies
    Ingredients: all-purpose flour, butter, white sugar, brown sugar, eggs, vanilla extract, baking soda, salt, chocolate chips
    User Request: "Can you make this recipe gluten-free and with less sugar?"
    Customized Recipe:
    Gluten-Free Lower-Sugar Chocolate Chip Cookies
    Ingredients: gluten-free flour blend, butter, coconut sugar (reduced amount), eggs, vanilla extract, baking soda, salt, dark chocolate chips
    Modifications:
    - Replaced all-purpose flour with gluten-free flour blend (add 1/2 tsp xanthan gum if not included in blend)
    - Reduced white and brown sugar by 25% and replaced with coconut sugar
    - Used dark chocolate chips with higher cocoa content (less sugar)
    - Added 1/4 tsp cinnamon to enhance sweetness perception without adding sugar
    """
    
    # Create the prompt for the Gemini model
    prompt = f"""
    {few_shot_examples}
    
    Original Recipe: {original_recipe["name"]}
    Ingredients: {", ".join(original_recipe["ingredients"])}
    Steps: {original_recipe["steps"]}
    User Request: "{user_request}"
    
    Provide a customized version of this recipe based on the user's request. Include:
    1. A suitable name for the customized recipe
    2. Modified list of ingredients
    3. Clear explanation of all modifications made
    4. Any special notes or tips for preparation
    
    Format your response as:
    Customized Recipe:
    [New Recipe Name]
    Ingredients:
    [Modified Ingredients List]
    Modifications:
    - [Modification 1]
    - [Modification 2]
    Special Notes:
    [Any special preparation notes]
    """
    
    # Use Gemini to generate the customized recipe
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    response = model.generate_content(prompt)
    
    # Parse the response text and structure it
    response_text = response.text
    
    result = {
        "original_recipe": original_recipe["name"],
        "original_id": recipe_id,
        "user_request": user_request,
        "customized_recipe": response_text
    }
    
    return result

# ============================================================
# PART 6: Visualization and Analysis
# ============================================================

def visualize_recipe_customization_metrics(conn):
    """Visualize metrics and insights about recipe customization patterns."""
    # For demonstration, we'll use our sample data to simulate customization metrics
    
    # 1. Most common dietary restrictions
    dietary_restrictions = ["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Low-Carb"]
    restriction_counts = [42, 35, 28, 22, 18]
    
    plt.figure(figsize=(10, 6))
    plt.bar(dietary_restrictions, restriction_counts, color='skyblue')
    plt.title('Most Common Dietary Restrictions in Recipe Customizations')
    plt.xlabel('Restriction Type')
    plt.ylabel('Number of Customizations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 2. Most commonly added/removed ingredients
    removed_ingredients = ["Chicken", "Beef", "Dairy", "Gluten", "Sugar"]
    removed_counts = [38, 32, 30, 25, 20]
    
    added_ingredients = ["Tofu", "Mushrooms", "Plant Milk", "GF Flour", "Nutritional Yeast"]
    added_counts = [35, 32, 28, 25, 22]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(removed_ingredients, removed_counts, color='salmon')
    plt.title('Most Commonly Removed Ingredients')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(added_ingredients, added_counts, color='lightgreen')
    plt.title('Most Commonly Added Ingredients')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Customization satisfaction scores (simulated)
    satisfaction_data = {
        'Vegetarian': [4.2, 4.5, 3.8, 4.7, 4.3],
        'Vegan': [4.0, 3.9, 4.2, 3.7, 4.1],
        'Gluten-Free': [3.7, 4.0, 3.5, 3.8, 4.2],
        'Low-Carb': [4.1, 3.9, 4.3, 4.0, 3.8]
    }
    
    plt.figure(figsize=(10, 6))
    plt.boxplot([satisfaction_data[k] for k in satisfaction_data.keys()], 
                labels=satisfaction_data.keys())
    plt.title('User Satisfaction Scores by Dietary Restriction Type')
    plt.ylabel('Satisfaction Score (1-5)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    return {
        "dietary_restrictions": dict(zip(dietary_restrictions, restriction_counts)),
        "removed_ingredients": dict(zip(removed_ingredients, removed_counts)),
        "added_ingredients": dict(zip(added_ingredients, added_counts)),
        "satisfaction_scores": satisfaction_data
    }

# ============================================================
# PART 7: Demo & Usage
# ============================================================

def run_recipe_customization_demo():
    """Run a demonstration of the recipe customization system."""
    print("# Recipe Customization System Demonstration")
    
    # Set up the databases
    db_conn = setup_sqlite_database()
    chroma_client, recipe_collection, user_collection = setup_vector_database()
    
    # Load sample data
    load_sample_data(db_conn, recipe_collection)
    
    print("\n## Database Inspection")
    tables = list_tables(db_conn)
    print(f"Tables in database: {', '.join(tables)}")
    
    # Show a sample recipe
    print("\n## Sample Recipe")
    sample_recipe = get_recipe_by_id(db_conn, 1)
    print(f"Recipe: {sample_recipe['name']}")
    print(f"Ingredients: {', '.join(sample_recipe['ingredients'])}")
    print(f"Cook time: {sample_recipe['minutes']} minutes")
    print(f"Steps: {len(sample_recipe['steps'])} steps")
    
    # Demonstrate recipe search
    print("\n## Recipe Search")
    print("Searching for 'vegetable' recipes...")
    vegetable_recipes = search_recipes_by_text(recipe_collection, "vegetable", 2)
    for recipe_id in vegetable_recipes:
        recipe = get_recipe_by_id(db_conn, recipe_id)
        print(f" - {recipe['name']} (ID: {recipe['id']})")
    
    # Demonstrate ingredient-based search
    print("\nSearching for recipes with 'pasta' and 'garlic'...")
    pasta_recipes = search_recipes_by_ingredients(db_conn, recipe_collection, ["pasta", "garlic"], 2)
    for recipe in pasta_recipes:
        print(f" - {recipe['name']} (ID: {recipe['id']})")
    
    # Demonstrate similar recipes
    print("\n## Similar Recipes")
    print(f"Finding recipes similar to '{sample_recipe['name']}'...")
    similar_recipes = get_similar_recipes(db_conn, recipe_collection, 1, "ingredients", 2)
    for recipe in similar_recipes:
        print(f" - {recipe['name']} (ID: {recipe['id']})")
    
    # Demonstrate recipe customization with function calling
    print("\n## Recipe Customization with Function Calling")
    print("Customizing Creamy Garlic Pasta to be vegan...")
    
    customization_result = customize_recipe(
        db_conn, 
        1, 
        dietary_restrictions=["vegan"],
        ingredients_to_remove=["heavy cream", "parmesan cheese"],
        ingredients_to_add=["coconut milk", "nutritional yeast"]
    )
    
    print(f"Customized Recipe: {customization_result['name']}")
    print(f"Ingredients: {', '.join(customization_result['ingredients'])}")
    print("Modifications:")
    for mod in customization_result['modifications']:
        print(f" - {mod}")
    
    # Demonstrate chat interface with function calling
    print("\n## Chat Interface with Function Calling")
    query = "I want to make a vegetarian version of the beef lasagna. Can you help me?"
    print(f"User Query: '{query}'")
    
    response = recipe_customization_chat(db_conn, recipe_collection, query)
    
    if response["function_called"]:
        print(f"Function Called: {response['function_called']}")
        print(f"AI Response: {response['assistant_response']}")
    else:
        print(f"AI Response: {response['assistant_response']}")
    
    # Demonstrate few-shot prompting
    print("\n## Few-Shot Prompting for Recipe Customization")
    request = "I'm on a low-carb diet. Can you adapt this pasta recipe for me?"
    print(f"User Request: '{request}'")
    
    few_shot_result = customize_recipe_with_few_shot(db_conn, 1, request)
    
    print("Customization Result:")
    print(few_shot_result["customized_recipe"])
    
    # Visualization demo
    print("\n## Recipe Customization Analytics")
    print("Generating visualizations of recipe customization patterns...")
    metrics = visualize_recipe_customization_metrics(db_conn)
    
    print("\n## Demo Complete!")
    print("The recipe customization system has demonstrated the following capabilities:")
    print(" - Hybrid storage with SQLite and vector embeddings")
    print(" - Semantic search for recipes")
    print(" - Function calling with the Gemini API")
    print(" - Few-shot prompting for recipe customization")
    print(" - Visualization of customization patterns")

if __name__ == "__main__":
    run_recipe_customization_demo()
