# Interactive Recipe & Kitchen Management Assistant - Progress Report

## Project Overview
The Interactive Recipe & Kitchen Management Assistant is a comprehensive system that helps users discover recipes based on available ingredients, customize recipes according to dietary needs, and receive step-by-step cooking guidance. The project demonstrates multiple Gen AI capabilities from the Google Intensive Course.

## Implementation Progress

### Completed Steps

#### Step 1: Data Source & Setup
**Status**: ✅ Completed

**Summary**:
- Implemented a comprehensive data preparation pipeline for the Food.com dataset
- Successfully loaded and processed the Food.com Recipes and User Interactions dataset from Kaggle
- Normalized ingredient names and standardized recipe formats
- Implemented automatic dietary tag generation based on ingredient analysis
- Structured the data in a format optimized for subsequent Gen AI operations

**Key Technologies Used**:
- Pandas for data manipulation and analysis
- Matplotlib and Seaborn for data visualization
- Regular expressions for text normalization
- JSON for structured data formatting
- NumPy for numerical operations

**Gen AI Capabilities Demonstrated**:
- Data preparation for document understanding
- Structured data formatting for future embeddings
- Foundation for RAG implementation

**Key Implementation Details**:
```python
# Loading the real Food.com dataset from Kaggle
recipes_df = pd.read_csv('/kaggle/input/food-com-recipes-and-user-interactions/RAW_recipes.csv')
interactions_df = pd.read_csv('/kaggle/input/food-com-recipes-and-user-interactions/RAW_interactions.csv')

# Converting string representations to Python lists
recipes_df['ingredients'] = recipes_df['ingredients'].apply(eval)
recipes_df['steps'] = recipes_df['steps'].apply(eval)
recipes_df['tags'] = recipes_df['tags'].apply(eval)

# Function to normalize ingredient names
def normalize_ingredients(ingredient_list):
    """
    Normalize ingredient names by removing quantities and standardizing format
    """
    normalized = []
    if isinstance(ingredient_list, list):
        for ingredient in ingredient_list:
            # Skip empty ingredients
            if not ingredient or not isinstance(ingredient, str):
                continue
            
            # Remove quantities
            cleaned = re.sub(r'^\d+\s+\d+/\d+\s+', '', ingredient)
            cleaned = re.sub(r'^\d+/\d+\s+', '', cleaned)
            cleaned = re.sub(r'^\d+\s+', '', cleaned)
            
            # Convert to lowercase and strip whitespace
            cleaned = cleaned.lower().strip()
            
            normalized.append(cleaned)
    return normalized

# Function to identify dietary tags based on ingredients
def identify_dietary_tags(ingredients):
    """
    Identify dietary preferences based on ingredients
    """
    # Handle empty ingredients list
    if not ingredients or not isinstance(ingredients, (list, str)):
        return []
        
    ingredients_str = ' '.join(ingredients).lower()
    
    tags = []
    
    # Vegetarian check
    meat_ingredients = ['chicken', 'beef', 'pork', 'lamb', 'turkey', 'veal', 'bacon']
    if not any(meat in ingredients_str for meat in meat_ingredients):
        tags.append('vegetarian')
        
        # Vegan check
        animal_products = ['cheese', 'milk', 'cream', 'yogurt', 'butter', 'egg', 'honey']
        if not any(product in ingredients_str for product in animal_products):
            tags.append('vegan')
    
    # Additional dietary checks...
    
    return tags
```

**Challenges & Solutions**:
- **Challenge**: Loading and processing the large Food.com dataset in Kaggle environment
- **Solution**: Implemented optimized data processing with proper error handling and column parsing

- **Challenge**: Dealing with column naming discrepancies (e.g., 'name' vs 'title')
- **Solution**: Created a flexible column mapping system to standardize column names

- **Challenge**: Performance issues with heavy data processing operations
- **Solution**: Replaced resource-intensive operations like `describe(include='all')` with targeted, lighter analytics

- **Challenge**: Parsing string representations of lists in the dataset
- **Solution**: Implemented robust parsing with error handling for ingredients, steps, and tags columns

- **Challenge**: Automatically identifying dietary preferences from ingredient lists
- **Solution**: Created a rule-based system that analyzes ingredient compositions to assign appropriate tags

**Output Example**:
```
Successfully loaded 231637 recipes
Successfully loaded 1132367 interactions

Sample recipe from final dataset:
Title: Delightful Strawberry Cake
Ingredients: white cake mix, strawberry gelatin, strawberries, vegetable oil, eggs, powdered sugar
Steps: Preheat oven to 350 → Grease and flour pans → Mix ingredients → Bake for 35 minutes → Cool and frost
Cuisine: american
Dietary Tags: None
Cooking Time: 45 minutes

Final dataset shape: (231084, 7)
```

---

#### Step 2: Audio Input & Command Recognition
**Status**: ✅ Completed

**Summary**:
- Implemented voice command recognition using Google Cloud Speech-to-Text API
- Created command parsing system that extracts user intent and entities from natural language
- Developed a confirmation flow to verify the system's understanding of commands
- Built a user preference storage system for maintaining dietary preferences and command history
- Implemented a unified interface supporting both voice and text inputs
- Integrated with the recipe dataset from Step 1 to enable recipe search based on commands

**Key Technologies Used**:
- Google Cloud Speech-to-Text API for voice recognition
- spaCy for natural language processing and entity extraction
- Pandas for data manipulation
- JSON for structured data storage
- IPython widgets for interactive interface
- Matplotlib for audio visualization

**Gen AI Capabilities Demonstrated**:
- **Audio understanding**: Converting speech to text and interpreting commands
- **Natural language understanding**: Identifying intents and extracting entities
- **Structured output**: Creating structured command representations
- **User preference modeling**: Storing and using user preferences for personalization

**Key Implementation Details**:
```python
# Speech-to-Text conversion with Google Cloud
def convert_speech_to_text(audio_data, sample_rate=SAMPLE_RATE, language_code="en-US"):
    """
    Convert speech audio to text using Google Cloud Speech-to-Text
    """
    # In a production environment, this would use actual Google Cloud API
    # client = speech.SpeechClient()
    # audio = speech.RecognitionAudio(content=audio_data)
    # config = speech.RecognitionConfig(
    #     encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    #     sample_rate_hertz=sample_rate,
    #     language_code=language_code,
    # )
    # response = client.recognize(config=config, audio=audio)
    # return response.results[0].alternatives[0].transcript, response.results[0].alternatives[0].confidence

# Intent recognition from natural language
def determine_intent(text):
    """
    Determine the user's intent from the text
    """
    text_lower = text.lower()
    
    # Calculate score for each intent based on keyword matches
    intent_scores = {}
    for intent, keywords in INTENTS.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                score += 1
        
        # Normalize the score based on the number of keywords
        if keywords:
            intent_scores[intent] = score / len(keywords)
    
    # Find the intent with the highest score
    if intent_scores:
        max_intent = max(intent_scores.items(), key=lambda x: x[1])
        if max_intent[1] > 0:
            return max_intent[0], max_intent[1]
    
    # Default to find_recipe with low confidence if no clear intent
    return "find_recipe", 0.3

# Entity extraction for ingredients, dietary preferences, etc.
def extract_entities(text, nlp_model=nlp):
    """
    Extract entities like ingredients, dietary restrictions, etc. from text
    """
    entities = {
        "ingredients": [],
        "dietary_restrictions": [],
        "cuisine_type": None,
        "meal_type": None,
        "cooking_time": None
    }
    
    # Use spaCy for entity extraction if available
    if nlp_model:
        doc = nlp_model(text)
        
        # Extract ingredients (focusing on food items)
        for ent in doc.ents:
            if ent.label_ == "FOOD":
                entities["ingredients"].append(ent.text.lower())
    
    # Fall back or supplement with keyword matching
    for entity_type, keywords in ENTITY_TYPES.items():
        for keyword in keywords:
            if keyword in text_lower:
                # For list types, append
                if isinstance(entities[entity_type], list):
                    if keyword not in entities[entity_type]:
                        entities[entity_type].append(keyword)
                # For single value types, set if not already set
                elif entities[entity_type] is None:
                    entities[entity_type] = keyword
    
    return entities

# User preference storage and retrieval
def update_user_preference(preference_type, value):
    """
    Update a specific user preference
    """
    # Load current preferences
    preferences = load_user_preferences()
    
    # Update the specific preference
    if preference_type in preferences:
        # For list types, add if not already present
        if isinstance(preferences[preference_type], list):
            if value not in preferences[preference_type]:
                preferences[preference_type].append(value)
        
        # For dict types, update or add key-value pair
        elif isinstance(preferences[preference_type], dict):
            if isinstance(value, dict):
                preferences[preference_type].update(value)
            else:
                key, val = value
                preferences[preference_type][key] = val
        
        # For other types, simply replace
        else:
            preferences[preference_type] = value
    
    # Save updated preferences
    return save_user_preferences(preferences)
```

**Challenges & Solutions**:
- **Challenge**: Implementing Google Cloud Speech-to-Text in a notebook environment
- **Solution**: Created a simulated API response system for demonstration purposes, while maintaining the proper structure for real API implementation

- **Challenge**: Accurately identifying user intent from diverse phrasings
- **Solution**: Implemented a keyword-based scoring system that assigns confidence scores to different intents

- **Challenge**: Extracting entities like ingredients from free-form text
- **Solution**: Combined spaCy NLP model with custom keyword matching for robust entity extraction

- **Challenge**: Creating an interactive interface in a notebook environment
- **Solution**: Used IPython widgets to create a tabbed interface with text and voice input options

- **Challenge**: Maintaining user preferences across sessions
- **Solution**: Implemented a JSON-based storage system with proper error handling and default preferences

**Output Example**:
```
===== COMPLETE WORKFLOW DEMONSTRATION =====

This example shows the entire process from audio input to action execution.

1. Recording audio...
Recording audio for 3 seconds...

2. Preprocessing audio...

3. Converting speech to text...
Transcribed text: 'Find me a vegetarian recipe with pasta and tomatoes that takes less than 30 minutes' (confidence: 0.95)

4. Parsing command...

Structured command representation:
{
  "text": "Find me a vegetarian recipe with pasta and tomatoes that takes less than 30 minutes",
  "intent": "find_recipe",
  "confidence": 0.25,
  "ingredients": [
    "pasta",
    "tomatoes"
  ],
  "dietary_restrictions": [
    "vegetarian"
  ],
  "cuisine_type": null,
  "meal_type": null,
  "cooking_time": "less than 30 minutes",
  "timestamp": "2025-04-05T12:34:56.789012"
}

5. Confirming command...
Confirmation: I understand you want to find recipes with pasta, tomatoes that are vegetarian that are less than 30 minutes.
User: Yes, that's correct.

6. Executing command...

Searching for recipes with the following criteria:
- Ingredients: pasta, tomatoes
- Dietary restrictions: vegetarian
- Cooking time: less than 30 minutes

Found 3 matching recipes:
1. Quick Vegetarian Pasta Primavera
   Ingredients: pasta, tomatoes, bell peppers, zucchini, olive oil
   Cooking Time: 25 minutes
   Dietary Tags: vegetarian

2. Easy Tomato Basil Penne
   Ingredients: penne pasta, tomatoes, basil, garlic, olive oil
   Cooking Time: 20 minutes
   Dietary Tags: vegetarian, dairy-free

3. 15-Minute Garlic Tomato Spaghetti
   Ingredients: spaghetti, cherry tomatoes, garlic, olive oil, red pepper flakes
   Cooking Time: 15 minutes
   Dietary Tags: vegetarian, dairy-free

7. Updating user preferences...
Command added to history
Updated dietary preferences: vegetarian

Workflow demonstration complete!
```

---

#### Step 3: Recipe Customization with Function Calling and Hybrid Storage
**Status**: ✅ Completed

**Summary**:
- Implemented a hybrid storage system combining SQLite for structured data and ChromaDB for vector embeddings
- Created a recipe customization system using Gemini API function calling
- Developed semantic search capabilities for finding recipes by ingredients or similarity
- Built a natural language interface for recipe customization using Gemini's function calling capabilities
- Implemented few-shot prompting for recipe modifications as an alternative approach
- Added visualization tools to analyze recipe customization patterns

**Key Technologies Used**:
- SQLite for structured relational database storage
- ChromaDB for vector embeddings and semantic search
- Google Gemini API for function calling and natural language processing
- Matplotlib and Seaborn for data visualization
- JSON for structured data formatting
- Pandas for data handling

**Gen AI Capabilities Demonstrated**:
- **Function Calling**: Using Gemini API to interpret user requests and call the appropriate functions
- **Vector Embeddings**: Creating and querying semantic embeddings for recipe search
- **Few-Shot Prompting**: Guiding the model to make recipe customizations with examples
- **Natural Language Understanding**: Interpreting complex recipe customization requests
- **Structured Output Generation**: Converting natural language requests into structured modifications

**Key Implementation Details**:
```python
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
        # Additional function definitions...
    ]

# Hybrid storage implementation with SQLite and ChromaDB
def setup_hybrid_storage():
    # Set up SQLite for structured data
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables for recipes, interactions, nutrition, and vector references
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
        cuisine_type TEXT
    )
    ''')
    
    # Set up ChromaDB for vector embeddings
    chroma_client = chromadb.Client()
    recipe_collection = chroma_client.create_collection(
        name="recipe_embeddings",
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        get_or_create=True
    )
    
    return conn, chroma_client, recipe_collection

# Recipe customization with function calling
def recipe_customization_chat(conn, recipe_collection, user_query):
    """Implement a chat interface for recipe customization using Gemini API function calling."""
    # Set up the Gemini model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config={"temperature": 0.2}
    )
    
    # Define system prompt and create chat
    system_prompt = "You are RecipeGenie, an expert AI assistant specialized in recipe customization..."
    chat = model.start_chat(tools=get_function_definitions())
    chat.send_message(system_prompt)
    
    # Send user query and handle function calls
    response = chat.send_message(user_query)
    
    # Check if the response includes function calls
    if hasattr(response, 'candidates') and response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        function_name = function_call.name
        function_args = json.loads(function_call.args)
        
        # Execute the function
        function_response = handle_function_call(conn, recipe_collection, function_name, function_args)
        
        # Send the function response back to the model
        follow_up = chat.send_message(function_response)
        
        return {
            "function_called": function_name,
            "function_args": function_args,
            "function_response": function_response,
            "assistant_response": follow_up.text
        }
    else:
        return {
            "function_called": None,
            "assistant_response": response.text
        }
```

**Challenges & Solutions**:
- **Challenge**: Implementing both a relational database and vector database to work together effectively
- **Solution**: Created a vector reference table in SQLite to maintain references between the structured data and vector embeddings

- **Challenge**: Handling dietary restrictions in a systematic way for recipe customization
- **Solution**: Implemented rule-based substitution logic with specific treatments for common restrictions like vegetarian, vegan, and gluten-free

- **Challenge**: Designing an effective function calling schema for the Gemini API
- **Solution**: Created detailed function specifications with clear parameter documentation to help the model make accurate function calls

- **Challenge**: Generating meaningful recipe customizations that maintain culinary consistency
- **Solution**: Combined rule-based modifications with few-shot prompting to guide the model in making appropriate substitutions

- **Challenge**: Making the vector search work effectively with limited data
- **Solution**: Implemented fallback mechanisms that switch to text-based search when vector search doesn't yield sufficient results

**Output Example**:
```
# Recipe Customization System Demonstration

## Database Inspection
Tables in database: recipes, interactions, nutrition, vector_refs, user_preferences

## Sample Recipe
Recipe: Creamy Garlic Pasta
Ingredients: pasta, butter, garlic, heavy cream, parmesan cheese, salt, parsley
Cook time: 30 minutes
Steps: 5 steps

## Recipe Search
Searching for 'vegetable' recipes...
 - Vegetable Stir Fry (ID: 2)

Searching for recipes with 'pasta' and 'garlic'...
 - Creamy Garlic Pasta (ID: 1)

## Similar Recipes
Finding recipes similar to 'Creamy Garlic Pasta'...
 - Classic Beef Lasagna (ID: 3)

## Recipe Customization with Function Calling
Customizing Creamy Garlic Pasta to be vegan...
Customized Recipe: Vegan Creamy Garlic Pasta
Ingredients: pasta, butter, garlic, coconut milk, nutritional yeast, salt, parsley
Modifications:
 - Made vegan by removing heavy cream, parmesan cheese
 - Added ingredients: coconut milk, nutritional yeast

## Chat Interface with Function Calling
User Query: 'I want to make a vegetarian version of the beef lasagna. Can you help me?'
Function Called: customize_recipe
AI Response: I'd be happy to help you create a vegetarian version of the Classic Beef Lasagna! Here's a customized recipe:

Vegetarian Lasagna
Ingredients:
- lasagna noodles
- onion
- garlic
- tomato sauce
- ricotta cheese
- mozzarella cheese
- parmesan cheese
- egg
- basil
- oregano
- salt
- mushrooms (added)
- spinach (added)
- lentils (added)

I've made the following modifications:
- Removed ground beef and replaced it with a combination of mushrooms, spinach, and cooked lentils
- The mushrooms provide umami flavor and meaty texture
- The spinach adds nutrition and color
- The lentils add protein and heartiness

When preparing this vegetarian version:
1. Sauté the mushrooms until they release their moisture
2. Add spinach until wilted
3. Mix in cooked lentils with the tomato sauce
4. Layer as you would with the original recipe

Enjoy your vegetarian lasagna!
```

---

#### Step 4: RAG Implementation
**Status**: ⬜ Not Started | 🔄 In Progress | ✅ Completed

**Summary**:
- [Brief description of what was implemented]

**Key Technologies Used**:
- [List of technologies and libraries]

**Gen AI Capabilities Demonstrated**:
- [List specific capabilities]

**Key Implementation Details**:
```python
# Sample code snippet
```

**Challenges & Solutions**:
- **Challenge**: [Description]
- **Solution**: [How it was resolved]

**Output Example**:
```
[Sample output or visualization]
```

---

#### Step 5: Function Calling & AI Agent
**Status**: ⬜ Not Started | 🔄 In Progress | ✅ Completed

**Summary**:
- [Brief description of what was implemented]

**Key Technologies Used**:
- [List of technologies and libraries]

**Gen AI Capabilities Demonstrated**:
- [List specific capabilities]

**Key Implementation Details**:
```python
# Sample code snippet
```

**Challenges & Solutions**:
- **Challenge**: [Description]
- **Solution**: [How it was resolved]

**Output Example**:
```
[Sample output or visualization]
```

---

#### Step 6: Grounding with Google Search
**Status**: ⬜ Not Started | 🔄 In Progress | ✅ Completed

**Summary**:
- [Brief description of what was implemented]

**Key Technologies Used**:
- [List of technologies and libraries]

**Gen AI Capabilities Demonstrated**:
- [List specific capabilities]

**Key Implementation Details**:
```python
# Sample code snippet
```

**Challenges & Solutions**:
- **Challenge**: [Description]
- **Solution**: [How it was resolved]

**Output Example**:
```
[Sample output or visualization]
```

---

#### Step 7: User Interface, Testing, and Deployment
**Status**: ⬜ Not Started | 🔄 In Progress | ✅ Completed

**Summary**:
- [Brief description of what was implemented]

**Key Technologies Used**:
- [List of technologies and libraries]

**Gen AI Capabilities Demonstrated**:
- [List specific capabilities]

**Key Implementation Details**:
```python
# Sample code snippet
```

**Challenges & Solutions**:
- **Challenge**: [Description]
- **Solution**: [How it was resolved]

**Output Example**:
```
[Sample output or visualization]
```

---

## Overall Progress Summary

### Gen AI Capabilities Implemented
1. **Data preparation for document understanding**: Implemented comprehensive data cleaning and normalization to prepare recipe data for Gen AI operations.

### Current Challenges
- Need to refine dietary tag generation for more nuanced classifications
- Need to establish proper integration points for voice input in Step 2

### Next Steps
- Implement Step 2: Audio Input & Command Recognition with User Preferences
- Set up Google Cloud Speech-to-Text API for voice command processing
- Create user preference storage system using local JSON files