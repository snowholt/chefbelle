import sqlite3
import json
import os
import requests
from typing import List, Tuple, Dict, Optional

# Path to SQL database
DB_PATH = "final/kitchen_db.sqlite"

def list_tables() -> List[str]:
    """List all tables in the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    return [table[0] for table in tables]

def describe_table(table_name: str) -> List[Tuple[str, str]]:
    """Describe the schema of a specified table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema = cursor.fetchall()
    conn.close()
    return [(col[1], col[2]) for col in schema]

def execute_query(sql: str) -> List[Tuple]:
    """Execute an SQL query and return the results."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results
    except sqlite3.Error as e:
        conn.close()
        # Return error message instead of the full error object for better handling downstream
        return [("Error executing SQL query:", str(e))] # Modified error return

# New function to fetch nutrition data from Open Food Facts API
def fetch_nutrition_from_openfoodfacts(ingredient_name: str) -> Dict:
    """Fetch nutrition data for an ingredient from Open Food Facts API."""
    api_key = os.getenv('OPENFOODFACTS_API_KEY')
    # It's good practice to check if the key exists, though Open Food Facts search might work without one for basic queries.
    # if not api_key:
    #     print("Warning: OPENFOODFACTS_API_KEY environment variable not set.")
    #     # Depending on strictness, you might return unavailable here or proceed without auth
    #     # return {"status": "unavailable", "reason": "API key not configured"}

    search_url = f"https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "search_terms": ingredient_name,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 1 # We only need the top result
    }
    headers = {'User-Agent': 'CapstoneProject/1.0 (Language Model Integration)'} # Good practice to set a User-Agent

    try:
        response = requests.get(search_url, params=params, headers=headers, timeout=10) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        data = response.json()

        if data.get('products') and len(data['products']) > 0:
            product = data['products'][0]
            nutriments = product.get('nutriments', {})

            # Extract desired nutrients, providing default None if not found
            # Using .get() avoids KeyError if a nutrient is missing
            nutrition_info = {
                "food_normalized": ingredient_name, # Keep track of what was searched
                "source": "Open Food Facts",
                "product_name": product.get('product_name', 'N/A'),
                "calories_100g": nutriments.get('energy-kcal_100g'),
                "fat_100g": nutriments.get('fat_100g'),
                "saturated_fat_100g": nutriments.get('saturated-fat_100g'),
                "carbohydrates_100g": nutriments.get('carbohydrates_100g'),
                "sugars_100g": nutriments.get('sugars_100g'),
                "fiber_100g": nutriments.get('fiber_100g'),
                "proteins_100g": nutriments.get('proteins_100g'),
                "sodium_100g": nutriments.get('sodium_100g'),
                # Add other relevant fields if needed, e.g., vitamins
            }
            # Filter out keys with None values for cleaner output
            return {k: v for k, v in nutrition_info.items() if v is not None}
        else:
            return {"status": "unavailable", "reason": "No product found on Open Food Facts"}

    except requests.exceptions.RequestException as e:
        print(f"Error fetching nutrition for '{ingredient_name}': {e}")
        return {"status": "unavailable", "reason": f"API request failed: {e}"}
    except json.JSONDecodeError:
        print(f"Error decoding JSON response for '{ingredient_name}'")
        return {"status": "unavailable", "reason": "Invalid JSON response from API"}


def get_recipe_by_id(recipe_id: str) -> Optional[dict]:
    """Get a recipe by its ID with all details, including live nutrition data for ingredients."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Use try-finally to ensure connection is closed
    try:
        cursor.execute("SELECT * FROM recipes WHERE id = ?", (recipe_id,))
        recipe_data = cursor.fetchone()

        if not recipe_data:
            return None

        columns = [col[0] for col in cursor.description]
        recipe = dict(zip(columns, recipe_data))

        # Standardize field parsing - attempt JSON, fallback for simple strings
        for field in ["steps", "ingredients", "nutrition", "tags", "dietary_tags", "normalized_ingredients"]:
            value = recipe.get(field)
            if isinstance(value, str):
                try:
                    recipe[field] = json.loads(value)
                except json.JSONDecodeError:
                    # If JSON fails, and it's one of the list-like fields, try splitting by space
                    if field in ["ingredients", "tags", "dietary_tags", "normalized_ingredients"]:
                         recipe[field] = value.split() # Split space-separated string into list
                         print(f"Info: Field '{field}' in recipe ID {recipe_id} was treated as space-separated string.")
                    # else: # Keep as string if it's not expected to be a list (like 'steps' or 'nutrition' if not JSON)
                    #    print(f"Warning: Could not parse JSON for field '{field}' in recipe ID {recipe_id}. Kept as string. Value: {value[:100]}...")
                    #    pass # Keep as string if parsing fails and it's not a simple list field
                    # Simplified: Just split the known list-like fields if JSON fails
                    elif field == "steps": # Keep steps as string if not JSON
                         print(f"Warning: Could not parse JSON for field 'steps' in recipe ID {recipe_id}. Kept as string.")
                         pass # Keep as string
                    else: # Handle nutrition or other fields if necessary
                         print(f"Warning: Could not parse JSON for field '{field}' in recipe ID {recipe_id}. Value: {value[:100]}...")
                         pass # Keep as string or handle differently if needed


        # Fetch nutrition for normalized ingredients
        ingredient_nutrition_list = []
        normalized_ingredients = recipe.get("normalized_ingredients")

        # Ensure normalized_ingredients is now a list before iterating
        if isinstance(normalized_ingredients, list):
            for ingredient in normalized_ingredients:
                if isinstance(ingredient, str): # Ensure it's a string before fetching
                     # Check for empty strings that might result from splitting
                     if ingredient.strip():
                         nutrition_data = fetch_nutrition_from_openfoodfacts(ingredient)
                         ingredient_nutrition_list.append(nutrition_data)
                else:
                     # Handle cases where items in the list aren't strings
                     ingredient_nutrition_list.append({"status": "unavailable", "reason": f"Invalid ingredient format: {type(ingredient)}"})
        # Removed the 'elif normalized_ingredients is not None:' block as the splitting logic above handles the string case.
        # If it's still not a list after the processing above, something else is wrong.
        elif normalized_ingredients is not None:
             print(f"Error: 'normalized_ingredients' field in recipe ID {recipe_id} is not a list after processing. Type: {type(normalized_ingredients)}")
             ingredient_nutrition_list.append({"status": "unavailable", "reason": "normalized_ingredients field could not be processed into a list"})


        recipe['ingredient_nutrition'] = ingredient_nutrition_list # Add the fetched nutrition data

        return recipe
    finally:
        conn.close() # Ensure connection is closed even if errors occur


def get_ratings_and_reviews_by_recipe_id(recipe_id: str, limit: int) -> Optional[dict]: # Removed default for limit
    """Get ratings and the most recent reviews for a recipe ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Use try-finally for connection closing
    try:
        # Get overall rating
        cursor.execute("SELECT AVG(rating) FROM interactions WHERE recipe_id = ?", (recipe_id,))
        overall_rating_result = cursor.fetchone()
        # Handle case where there are no ratings yet
        overall_rating = overall_rating_result[0] if overall_rating_result else None

        # Get most recent reviews
        cursor.execute(
            "SELECT date, rating, review FROM interactions WHERE recipe_id = ? ORDER BY date DESC LIMIT ?",
            (recipe_id, limit),
        )
        recent_reviews = cursor.fetchall()
        columns = ["date", "rating", "review"]
        reviews_list = [dict(zip(columns, review)) for review in recent_reviews]

        return {"overall_rating": overall_rating, "recent_reviews": reviews_list}
    finally:
        conn.close()

# These are the Python functions defined above.
db_tools = [list_tables, describe_table, execute_query, get_ratings_and_reviews_by_recipe_id, get_recipe_by_id, fetch_nutrition_from_openfoodfacts]

instruction = """You are a helpful chatbot that can interact with an SQL database for a Kitchen Assistant.
You can retrieve information about recipes and user interactions (ratings and reviews).
Use the available tools to understand the database schema and execute SQL queries to answer user questions.
**Note:** Ingredient nutrition information is fetched live from the Open Food Facts API when you request recipe details using `get_recipe_by_id`.

Here are the available tools:

- list_tables(): Lists all tables in the database.
- describe_table(table_name: str): Describes the schema (columns and their types) of a specified table.
- execute_query(sql: str): Executes an SQL query and returns the results. Use this for complex data retrieval. Be careful with the SQL syntax.
- get_recipe_by_id(recipe_id: str): Retrieves all details for a specific recipe given its ID. This includes fetching live nutrition data for each normalized ingredient from Open Food Facts.
- get_ratings_and_reviews_by_recipe_id(recipe_id: str, limit: int): Retrieves the overall rating and the 'limit' most recent reviews for a given recipe ID. You MUST provide a value for 'limit'. If the user doesn't specify, use 3.

When a user asks a question:

1. Understand the user's intent and identify which tool(s) can best answer the question.
2. If the question requires specific information about a recipe (like ingredients, steps, nutrition), use `get_recipe_by_id`. This will automatically include nutrition details fetched from Open Food Facts.
3. If the question asks for ratings or reviews for a recipe, use `get_ratings_and_reviews_by_recipe_id`. **Always specify a value for the 'limit' parameter. If the user doesn't say how many reviews, use 3.**
4. For more complex queries involving filtering, joining tables, or specific data selection, construct and use the `execute_query` tool. First, use `list_tables` and `describe_table` to understand the database structure if needed.
5. Present the information clearly and concisely to the user, ideally in a readable format like Markdown.

For example:

- To get details about recipe ID 71373 (including ingredient nutrition), use `get_recipe_by_id(recipe_id='71373')`.
- To get the overall rating and 3 recent reviews for recipe ID 71373, use `get_ratings_and_reviews_by_recipe_id(recipe_id='71373', limit=3)`.
- For a complex query like "Find recipes with 'chicken' and a cooking time less than 30 minutes", you might need to use `execute_query` after understanding the 'recipes' table schema.

Be mindful of potential SQL injection if using `execute_query` with user-provided input, although in this setup, the queries are constructed by you based on the user's intent.
"""



# These are the Python functions defined above.


client = genai.Client(api_key=GOOGLE_API_KEY)

# Start a chat with automatic function calling enabled.
chat = client.chats.create(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=instruction,
        tools=db_tools,
    ),
)



resp = chat.send_message("""check recipe id=71373 in both tables (recipes and interactions) , 
check both tables please, and return full info, start with recipes table,
then check the nutritions by calling the fetch_nutrition_from_openfoodfacts function and get the accumulated info of Ingredients inside of the recipe,
then interactions table, and get the info of that recipe_id, however show only overal rating for that recipe, and 3 most recent reviews not all of them. 
output in markdown format""")
display(Markdown(resp.text))