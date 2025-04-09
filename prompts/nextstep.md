# Next Implementation Step: Recipe Customization with Function Calling and Hybrid Storage

## Overview
This document outlines the implementation details for Step 3 of the Interactive Recipe & Kitchen Management Assistant project. The focus is on building a robust recipe customization system using function calling with the Gemini API and a hybrid storage approach combining SQLite for structured data and a vector database for embeddings.

## Project Context

Our dataset consists of:
- **231,637 recipes** with detailed attributes including ingredients, steps, and nutritional information
- **1,132,367 user interactions** capturing ratings and reviews
- **3,454 nutrition records** for different foods
- **Vectorized recipe data** (178,265 records) with token embeddings for semantic search
- **Vectorized user data** (25,076 records) capturing preferences and interaction history

## Implementation Plan

### 1. Hybrid Database Architecture

We'll implement a two-part storage solution:

1. **SQLite Database**: For relational data and structured queries
   - Core recipe information (name, ingredients list, steps, nutrition facts, etc.)
   - User interaction data (ratings, reviews)
   - Nutrition reference data

2. **Vector Store**: For embedding vectors to support semantic search
   - Name, ingredient and step embeddings
   - User preference embeddings
   - Techniques and other vectorized features

This hybrid approach leverages the strengths of both systems - SQL for structured querying and Vector DB for semantic similarity search in RAG workflows.

### 2. Database Setup in Kaggle

```python
# Load SQL extension for interactive queries
%load_ext sql
%sql sqlite:///recipes_db.sqlite

# Create tables with appropriate schema
%%sql
-- Main recipes table with core information
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
    cuisine_type TEXT
);

-- Interactions table for user ratings and reviews
CREATE TABLE IF NOT EXISTS interactions (
    user_id INTEGER,
    recipe_id INTEGER,
    date TEXT,
    rating INTEGER,
    review TEXT,
    PRIMARY KEY (user_id, recipe_id),
    FOREIGN KEY (recipe_id) REFERENCES recipes(id)
);

-- Nutrition facts table
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
    vitamin_b11 REAL
);

-- Vector reference table (storing references to vector DB)
CREATE TABLE IF NOT EXISTS vector_refs (
    id INTEGER PRIMARY KEY,
    table_name TEXT NOT NULL,
    record_id INTEGER NOT NULL,
    vector_id TEXT NOT NULL,
    vector_type TEXT NOT NULL
);
```

### 3. Vector Storage Implementation

For the vector embeddings, we'll use ChromaDB, a lightweight vector database that works well in Kaggle environments:

```python
# Install ChromaDB
!pip install chromadb

import chromadb
from chromadb.utils import embedding_functions

# Initialize client
chroma_client = chromadb.Client()

# Create recipe embeddings collection
recipe_emb_fn = embedding_functions.DefaultEmbeddingFunction()
recipe_collection = chroma_client.create_collection(
    name="recipe_embeddings",
    embedding_function=recipe_emb_fn
)

# Store recipe embeddings
for idx, row in pp_recipes.iterrows():
    recipe_id = row['id']
    
    # Store name embeddings
    if isinstance(row['name_tokens'], list) and len(row['name_tokens']) > 0:
        recipe_collection.add(
            ids=f"name_{recipe_id}",
            metadatas={"recipe_id": recipe_id, "type": "name"},
            embeddings=row['name_tokens']
        )
    
    # Store ingredient embeddings
    if isinstance(row['ingredient_tokens'], list) and len(row['ingredient_tokens']) > 0:
        recipe_collection.add(
            ids=f"ingredients_{recipe_id}",
            metadatas={"recipe_id": recipe_id, "type": "ingredients"},
            embeddings=row['ingredient_tokens']
        )
    
    # Store steps embeddings (techniques)
    if isinstance(row['steps_tokens'], list) and len(row['steps_tokens']) > 0:
        recipe_collection.add(
            ids=f"steps_{recipe_id}",
            metadatas={"recipe_id": recipe_id, "type": "steps"},
            embeddings=row['steps_tokens']
        )

# Create user embeddings collection
user_collection = chroma_client.create_collection(
    name="user_embeddings",
    embedding_function=recipe_emb_fn
)

# Store user embeddings (preferences)
for idx, row in pp_users.iterrows():
    user_id = row['u']
    
    # Store techniques preferences
    if isinstance(row['techniques'], list) and len(row['techniques']) > 0:
        user_collection.add(
            ids=f"techniques_{user_id}",
            metadatas={"user_id": user_id, "type": "techniques"},
            embeddings=row['techniques']
        )
```

### 4. Function Calling with Gemini API and LangGraph Integration

We'll implement a comprehensive set of functions for the Gemini API to call, integrating them with LangGraph for agent-based workflows. This approach enables complex recipe customization operations through a series of coordinated function calls.

```python
import sqlite3
import chromadb
import json
from typing import List, Dict, Any, Tuple, Optional
from google.generativeai import GenerativeModel
import langgraph.graph as lg
from langgraph.graph import END, StateGraph

# Database connection
db_file = "recipes_db.sqlite"
db_conn = sqlite3.connect(db_file)

# ChromaDB connection
chroma_client = chromadb.Client()
recipe_collection = chroma_client.get_collection("recipe_embeddings")
user_collection = chroma_client.get_collection("user_embeddings")

# ------- Core Database Functions -------

def list_tables() -> List[str]:
    """Retrieve the names of all tables in the database."""
    print(' - DB CALL: list_tables()')
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [t[0] for t in tables]

def describe_table(table_name: str) -> List[Tuple[str, str]]:
    """Look up the table schema.
    
    Args:
        table_name: Name of the table to describe
        
    Returns:
        List of columns, where each entry is a tuple of (column, type).
    """
    print(f' - DB CALL: describe_table({table_name})')
    cursor = db_conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema = cursor.fetchall()
    return [(col[1], col[2]) for col in schema]

def execute_query(sql: str) -> List[Any]:
    """Execute an SQL statement, returning the results.
    
    Args:
        sql: Valid SQL query to execute
        
    Returns:
        List of result rows
    """
    print(f' - DB CALL: execute_query({sql})')
    cursor = db_conn.cursor()
    cursor.execute(sql)
    return cursor.fetchall()

# ------- Recipe Customization Functions -------

def search_recipes_by_ingredients(ingredients: List[str], limit: int = 5) -> List[Dict[str, Any]]:
    """Search for recipes containing all or most of the specified ingredients.
    
    Args:
        ingredients: List of ingredient names to search for
        limit: Maximum number of recipes to return
        
    Returns:
        List of matching recipes with basic information
    """
    print(f' - RECIPE SEARCH: search_recipes_by_ingredients({ingredients}, {limit})')
    
    # SQL query to find recipes with matching ingredients
    placeholders = ', '.join(['?'] * len(ingredients))
    query = f"""
    SELECT r.id, r.name, r.n_ingredients, r.ingredients
    FROM recipes r
    WHERE r.id IN (
        SELECT recipe_id 
        FROM recipe_ingredients 
        WHERE ingredient_name IN ({placeholders})
        GROUP BY recipe_id
        HAVING COUNT(DISTINCT ingredient_name) >= {max(1, len(ingredients) // 2)}
    )
    LIMIT {limit}
    """
    
    cursor = db_conn.cursor()
    cursor.execute(query, ingredients)
    matches = cursor.fetchall()
    
    result = []
    for match in matches:
        recipe_id, name, n_ingredients, ingredients_json = match
        result.append({
            "id": recipe_id,
            "name": name,
            "n_ingredients": n_ingredients,
            "ingredients": json.loads(ingredients_json)
        })
    
    return result

def get_similar_recipes(recipe_id: int, similarity_type: str = "ingredients", limit: int = 5) -> List[Dict[str, Any]]:
    """Find recipes similar to the specified recipe based on the similarity type.
    
    Args:
        recipe_id: ID of the recipe to find similar ones for
        similarity_type: Type of similarity to use ('ingredients', 'techniques', or 'overall')
        limit: Maximum number of similar recipes to return
        
    Returns:
        List of similar recipes with basic information and similarity score
    """
    print(f' - SIMILARITY SEARCH: get_similar_recipes({recipe_id}, {similarity_type}, {limit})')
    
    # Get embedding ID for the recipe
    vector_id = f"{similarity_type}_{recipe_id}"
    
    # Query the vector database for similar recipes
    results = recipe_collection.query(
        id=vector_id,
        n_results=limit
    )
    
    # Get the IDs of similar recipes
    similar_recipe_ids = []
    for metadata in results['metadatas']:
        if 'recipe_id' in metadata and metadata['recipe_id'] != recipe_id:
            similar_recipe_ids.append(metadata['recipe_id'])
    
    # Get recipe details from SQL database
    if similar_recipe_ids:
        placeholders = ', '.join(['?'] * len(similar_recipe_ids))
        query = f"""
        SELECT id, name, n_ingredients, ingredients 
        FROM recipes 
        WHERE id IN ({placeholders})
        """
        
        cursor = db_conn.cursor()
        cursor.execute(query, similar_recipe_ids)
        matches = cursor.fetchall()
        
        result = []
        for match in matches:
            recipe_id, name, n_ingredients, ingredients_json = match
            result.append({
                "id": recipe_id,
                "name": name,
                "n_ingredients": n_ingredients,
                "ingredients": json.loads(ingredients_json)
            })
        
        return result
    
    return []

def customize_recipe(recipe_id: int, modifications: Dict[str, Any]) -> Dict[str, Any]:
    """Customize a recipe based on specified modifications.
    
    Args:
        recipe_id: ID of the recipe to customize
        modifications: Dictionary of modifications to apply, can include:
                      - dietary_restrictions: List of restrictions (e.g., 'vegetarian', 'gluten-free')
                      - ingredients_to_add: List of ingredients to add
                      - ingredients_to_remove: List of ingredients to remove
                      - serving_size: New serving size
                      - cooking_method: Alternative cooking method
    
    Returns:
        Modified recipe with applied changes
    """
    print(f' - RECIPE CUSTOMIZATION: customize_recipe({recipe_id}, {modifications})')
    
    # Get the original recipe
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM recipes WHERE id = ?", (recipe_id,))
    recipe_data = cursor.fetchone()
    
    if not recipe_data:
        return {"error": "Recipe not found"}
    
    # Convert to dictionary
    columns = [col[0] for col in cursor.description]
    recipe = dict(zip(columns, recipe_data))
    
    # Parse JSON fields
    recipe['ingredients'] = json.loads(recipe['ingredients'])
    recipe['steps'] = json.loads(recipe['steps'])
    
    # Apply modifications
    modified_recipe = recipe.copy()
    
    # Handle ingredient modifications
    if 'ingredients_to_remove' in modifications:
        modified_recipe['ingredients'] = [
            ing for ing in recipe['ingredients'] 
            if not any(remove_ing.lower() in ing.lower() for remove_ing in modifications['ingredients_to_remove'])
        ]
    
    if 'ingredients_to_add' in modifications:
        modified_recipe['ingredients'].extend(modifications['ingredients_to_add'])
    
    # Adjust serving size if needed
    if 'serving_size' in modifications:
        original_size = recipe.get('serving_size', 4)  # Assume default of 4 if not specified
        new_size = modifications['serving_size']
        ratio = new_size / original_size
        
        # Adjust ingredient quantities (simplified approach)
        # In a real implementation, this would need more sophisticated quantity parsing
        modified_recipe['serving_size'] = new_size
    
    # Modify cooking method if specified
    if 'cooking_method' in modifications:
        new_method = modifications['cooking_method']
        # Adapt cooking steps based on new method
        # This is a simplified approach; a real implementation would be more sophisticated
        modified_recipe['steps'] = [
            f"Using {new_method} instead: {step}" if "cook" in step.lower() else step
            for step in recipe['steps']
        ]
    
    return modified_recipe

# ------- LangGraph Agent Setup -------

def setup_recipe_agent():
    """Set up a LangGraph agent for recipe customization."""
    
    # Define the agent state
    class RecipeAgentState(lg.State):
        user_query: str
        current_recipes: Optional[List[Dict[str, Any]]] = None
        selected_recipe: Optional[Dict[str, Any]] = None
        customization_options: Optional[Dict[str, Any]] = None
        final_recipe: Optional[Dict[str, Any]] = None
        
    # Initialize Gemini model
    model = GenerativeModel("gemini-1.0-pro")
    
    # Define the nodes in our graph
    def analyze_query(state):
        """Analyze the user query to understand what they're looking for."""
        response = model.generate_content(
            f"Analyze this recipe query and extract key information: {state.user_query}. "
            "Extract ingredients, dish type, dietary restrictions, and cooking preferences."
        )
        analysis = response.text
        return {"analysis": analysis}
    
    def search_for_recipes(state):
        """Search for recipes based on the query analysis."""
        # Extract ingredients from analysis
        ingredients = ["..."]  # This would be extracted from the analysis
        recipes = search_recipes_by_ingredients(ingredients)
        return {"current_recipes": recipes}
    
    def select_recipe(state):
        """Select the best matching recipe based on the user's query."""
        if not state.current_recipes:
            return {"selected_recipe": None}
        
        # For simplicity, select the first recipe
        # In a real implementation, this would use more sophisticated matching
        return {"selected_recipe": state.current_recipes[0]}
    
    def determine_customizations(state):
        """Determine what customizations to make to the recipe."""
        response = model.generate_content(
            f"Based on this user query: '{state.user_query}', "
            f"and this recipe: {state.selected_recipe}, "
            f"what customizations should be made? "
            f"Consider dietary restrictions, ingredient substitutions, "
            f"cooking methods, and serving size adjustments."
        )
        
        # Parse the customizations
        customizations = {
            "dietary_restrictions": ["vegetarian"],  # Example
            "ingredients_to_remove": ["..."],
            "ingredients_to_add": ["..."],
            "serving_size": 2,
            "cooking_method": "..."
        }
        
        return {"customization_options": customizations}
    
    def apply_customizations(state):
        """Apply the customizations to the selected recipe."""
        if not state.selected_recipe or not state.customization_options:
            return {"final_recipe": None}
        
        customized_recipe = customize_recipe(
            state.selected_recipe["id"],
            state.customization_options
        )
        
        return {"final_recipe": customized_recipe}
    
    def generate_response(state):
        """Generate a natural language response with the customized recipe."""
        if not state.final_recipe:
            return "I couldn't find a suitable recipe based on your request."
        
        response = model.generate_content(
            f"Create a friendly response for this user query: '{state.user_query}'. "
            f"Include this customized recipe: {state.final_recipe}. "
            f"Format the recipe nicely with ingredients and steps."
        )
        
        return response.text
    
    # Define the graph
    workflow = StateGraph(RecipeAgentState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("search_for_recipes", search_for_recipes)
    workflow.add_node("select_recipe", select_recipe)
    workflow.add_node("determine_customizations", determine_customizations)
    workflow.add_node("apply_customizations", apply_customizations)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges
    workflow.add_edge("analyze_query", "search_for_recipes")
    workflow.add_edge("search_for_recipes", "select_recipe")
    workflow.add_edge("select_recipe", "determine_customizations")
    workflow.add_edge("determine_customizations", "apply_customizations")
    workflow.add_edge("apply_customizations", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Set the entry point
    workflow.set_entry_point("analyze_query")
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# Sample usage of the LangGraph agent
recipe_agent = setup_recipe_agent()

# Example query
result = recipe_agent.invoke({
    "user_query": "I want to make a vegetarian pasta dish with mushrooms and spinach, but I'm allergic to dairy."
})
```

### 5. Few-Shot Prompting Examples

To enhance the customization quality, we'll implement few-shot prompting with examples of recipe modifications:

```python
# Few-shot examples for recipe customization
customization_examples = """
Example 1:
Original Recipe: Chicken Alfredo Pasta
User Request: "I want to make this vegetarian and without dairy."
Modifications:
- Replace chicken with tofu or mushrooms
- Replace Alfredo sauce with olive oil, garlic and vegetable broth
- Add nutritional yeast for cheesy flavor without dairy
- Increase vegetables like spinach for added nutrition

Example 2:
Original Recipe: Beef Tacos
User Request: "I need this to be keto-friendly."
Modifications:
- Use lettuce wraps instead of tortillas
- Keep the beef but ensure it's higher fat content
- Add avocado and sour cream for healthy fats
- Reduce or eliminate beans and corn
- Include more low-carb vegetables like bell peppers

Example 3:
Original Recipe: Chocolate Chip Cookies
User Request: "I need this to be gluten-free and lower in sugar."
Modifications:
- Replace all-purpose flour with almond flour or gluten-free flour blend
- Reduce sugar by 30% and add some stevia or monk fruit sweetener
- Use dark chocolate chips with higher cocoa percentage
- Add cinnamon to enhance sweetness perception
- Include vanilla extract to enhance flavor without sugar
"""

def customize_recipe_with_few_shot(recipe_id: int, user_request: str) -> Dict[str, Any]:
    """Customize a recipe based on user request using few-shot prompting.
    
    Args:
        recipe_id: ID of the recipe to customize
        user_request: Natural language request for customization
        
    Returns:
        Modified recipe with applied changes
    """
    # Get the original recipe
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM recipes WHERE id = ?", (recipe_id,))
    recipe_data = cursor.fetchone()
    
    if not recipe_data:
        return {"error": "Recipe not found"}
    
    # Convert to dictionary
    columns = [col[0] for col in cursor.description]
    recipe = dict(zip(columns, recipe_data))
    
    # Create the few-shot prompt
    prompt = f"""
    {customization_examples}
    
    Original Recipe: {recipe['name']}
    Ingredients: {recipe['ingredients']}
    User Request: "{user_request}"
    
    Modifications:
    """
    
    # Generate customization ideas with Gemini
    model = GenerativeModel("gemini-1.0-pro")
    response = model.generate_content(prompt)
    
    # Parse the modifications and apply them
    # (In a real implementation, this would involve more sophisticated parsing)
    modifications = response.text.split('\n- ')
    modifications = [mod.strip() for mod in modifications if mod.strip()]
    
    # Apply the modifications to the recipe
    # (This is simplified; a real implementation would parse and apply each modification)
    modified_recipe = recipe.copy()
    modified_recipe['customization_notes'] = modifications
    
    return modified_recipe
```

### 6. Testing and Evaluation

To ensure our implementation works as expected, we'll set up comprehensive testing functions:

```python
def test_recipe_customization_pipeline():
    """Test the entire recipe customization pipeline with different queries."""
    test_queries = [
        "I want to make a vegetarian version of beef stroganoff",
        "I need a gluten-free chocolate cake recipe",
        "Show me how to make spaghetti bolognese without meat",
        "I'm on keto, can you adapt a mac and cheese recipe for me?"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: {query}")
        
        # Initialize the agent
        agent = setup_recipe_agent()
        
        # Process the query
        result = agent.invoke({"user_query": query})
        
        # Display the result
        print(f"Recipe Found: {result.selected_recipe['name'] if result.selected_recipe else 'None'}")
        if result.final_recipe:
            print(f"Customizations applied: {len(result.customization_options)}")
            print(f"Final recipe: {result.final_recipe['name']}")
    
    print("\nAll tests completed!")

# Run the tests
test_recipe_customization_pipeline()
```

## Next Steps

After implementing this solution:

1. **Optimize Performance**: Tune database indices and query performance for large datasets
2. **Enhance Customization Logic**: Add more sophisticated ingredient substitution rules
3. **User Feedback Loop**: Implement a mechanism to learn from user feedback on customized recipes
4. **Integration**: Connect this backend with the voice interface from Step 2
5. **Model Fine-Tuning**: Consider fine-tuning a model specifically for recipe customization tasks