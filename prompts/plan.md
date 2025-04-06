# INTERACTIVE RECIPE & KITCHEN MANAGEMENT ASSISTANT - IMPLEMENTATION PLAN

## STEP 1: DATA SOURCE & SETUP
Description:
- Use the Food.com Recipes and Interactions dataset from Kaggle, which has extensive recipe data including ingredients, steps, and ratings.
- Create a notebook for dataset exploration and cleaning to serve as the foundation for our recipe retrieval and customization system.
- Add dietary preference tags (vegan, gluten-free, low-sodium, etc.) for better categorization.

Task Requirements:
- Load and explore the Food.com dataset, focusing on recipe structure, ingredient lists, and preparation steps.
- Clean the data by removing duplicates, normalizing ingredient names, and standardizing measurements.
- Structure the data in pandas DataFrame format with columns: [recipe_id, title, ingredients, steps, cuisine_type, dietary_tags, cooking_time].

Prompt:
```
Load the Food.com Recipes dataset and perform the following:
1. Remove duplicate recipes and normalize ingredient names
2. Extract and standardize cooking times and measurements
3. Create dietary tags based on ingredient analysis (vegan, vegetarian, gluten-free, etc.)
4. Structure as a clean DataFrame with columns: [recipe_id, title, ingredients, steps, cuisine_type, dietary_tags, cooking_time]
```

---

## STEP 2: AUDIO INPUT & COMMAND RECOGNITION WITH USER PREFERENCES
Description:
- Implement speech-to-text functionality using Google Cloud Speech-to-Text API for voice commands.
- Add a text input alternative for users who prefer typing.
- Create a simple user preference storage system using local storage (dictionary/DataFrame) to remember dietary preferences, favorite recipes, and common substitutions.
- Implement a confirmation system for voice inputs to verify understanding before processing.

Task Requirements:
- Set up Google Cloud Speech-to-Text API integration for audio processing.
- Create a function to parse and validate user commands from both voice and text inputs.
- Implement a local storage mechanism using Python dictionaries saved as JSON files.
- Build a user confirmation flow that repeats back recognized commands and allows for corrections.

Prompt:
```
Initialize audio processing with Google Cloud Speech-to-Text:
- Convert audio input to text 
- Implement a confirmation function that repeats back the understood command
- Store user preferences in a local dictionary that persists between sessions
- Parse the command to identify: action type (find recipe, list ingredients, etc.), ingredients mentioned, and any dietary restrictions
```

---

## STEP 3: FEW-SHOT PROMPTING FOR RECIPE CUSTOMIZATION
Description:
- Use few-shot prompting to help the system understand various ways users might request recipe modifications.
- Create example templates for common recipe customization scenarios (dietary restrictions, ingredient substitutions, cooking method changes, etc.).
- Implement structured output format for recipe customization results.

Task Requirements:
- Develop a set of 5-7 few-shot examples covering different customization scenarios.
- Structure the examples to demonstrate input variations and expected outputs.
- Create a JSON schema for recipe customization output including modified ingredients, steps, and cooking times.
- Reference techniques from "day-1-prompting.ipynb" for effective few-shot implementation.

Prompt:
```
Create a few-shot prompt system with these example pairs:
1. "I need a gluten-free version of this pasta recipe" → [structured output showing pasta alternatives]
2. "Make this recipe low-sodium but still flavorful" → [structured output with salt alternatives]
3. "I don't have eggs, what can I substitute?" → [structured output with egg replacements]
4. "Convert this to an air fryer recipe" → [structured output with modified cooking method]
5. "Make this recipe for 8 people instead of 4" → [structured output with doubled quantities]

Return customized recipe in JSON format with modified ingredients, steps, and cooking time.
```

---

## STEP 4: RAG IMPLEMENTATION FOR RECIPE RETRIEVAL AND QUESTION ANSWERING
Description:
- Create a vector database using Chroma (from day-2-document-q-a-with-rag.ipynb) to store recipe texts, instructions, and cooking tips.
- Implement retrieval functionality to find relevant recipe chunks based on user queries.
- Enable natural language question answering about cooking techniques, ingredient substitutions, and recipe modifications.

Task Requirements:
- Generate embeddings for all recipes using Google's text-embedding-004 model.
- Chunk recipe instructions appropriately for effective retrieval.
- Implement similarity search using Chroma to retrieve relevant recipes and cooking instructions.
- Create a response generation system that combines retrieved information with the original query.

Prompt:
```
For RAG implementation:
1. Convert all recipes into embeddings using Google's text-embedding-004 model
2. Store these embeddings in a Chroma vector database
3. When a user asks a question, embed their query and retrieve the most relevant recipe chunks
4. Generate a comprehensive answer that combines retrieved information with the question context
```

---

## STEP 5: FUNCTION CALLING & AI AGENT
Description:
- Implement a set of specialized functions that the AI can call to perform specific tasks in the recipe management process.
- Create an agent workflow that can handle multi-step recipe tasks and follow a decision tree for complex requests.
- Enable automatic function calling based on user intent recognition.

Task Requirements:
- Define the following functions:
  * `find_recipe(ingredients: list, dietary_preferences: list, cuisine_type: str = None) -> list[dict]`
  * `get_recipe_details(recipe_id: str) -> dict`
  * `adjust_serving_size(recipe: dict, servings: int) -> dict`
  * `find_substitutes(ingredient: str, dietary_restrictions: list = None) -> list[str]`
  * `calculate_nutrition(recipe: dict) -> dict`
  * `save_user_preference(preference_type: str, value: str) -> bool`
- Create an agent workflow that chains these functions based on user needs.
- Implement error handling and recovery for failed function calls.

Prompt:
```
Implement function declarations following the Gemini API function calling pattern:
- Design each function to handle a specific recipe task
- Create a decision tree for the agent to determine which functions to call based on user requests
- Ensure proper error handling and allow the agent to recover from failed function calls
- Return results in a consistent JSON format
```

---

## STEP 6: GROUNDING WITH GOOGLE SEARCH INTEGRATION
Description:
- Integrate Google Search capabilities from the Gemini API to ground responses in factual information.
- Enable the system to search for recipes, cooking techniques, or ingredient information that's not in the local database.
- Present search results with proper attribution and citations.

Task Requirements:
- Implement the Google Search tool from the Gemini API (day-4-google-search-grounding.ipynb).
- Define trigger conditions for when to use external search (unknown recipes, celebrity chef references, etc.).
- Create a presentation format for search results that includes sources and relevant snippets.
- Ensure search results are properly integrated with locally stored information.

Prompt:
```
Enable grounding with Google Search:
- Trigger search when encountering unknown recipes, celebrity chef questions, or nutrition facts
- Format search results with proper attribution to sources
- Combine search results with the local recipe database information
- Present a unified response that clearly distinguishes between stored knowledge and externally retrieved information
```

---

## STEP 7: USER INTERFACE, TESTING, AND DEPLOYMENT
Description:
- Combine all components into a cohesive user experience with both voice and text inputs.
- Implement a structured output system that provides recipes in both JSON format and human-readable form.
- Create visualization components for recipes, including ingredient proportions and step-by-step instructions.
- Test the system with a variety of user scenarios and edge cases.

Task Requirements:
- Create a simple user interface within the Kaggle notebook environment.
- Implement JSON output for recipe instructions with step-by-step details.
- Add visualization for recipe components (ingredient charts, cooking time visualizations).
- Develop a test suite with sample user interactions covering both typical and edge cases.
- Document the entire system architecture and explain all Gen AI capabilities used.

Prompt:
```
Final system integration:
1. Combine all components into a single coherent flow
2. Create sample user interaction flows demonstrating the complete path from user query to recipe output
3. Generate structured JSON output for machine readability and formatted text for human consumption
4. Add basic visualizations for recipe components
5. Document the architecture, highlighting the Gen AI capabilities used (Audio understanding, Few-shot prompting, Function calling, RAG, Grounding)
```

## IMPLEMENTATION NOTES

### Cuisine and Dietary Support
- The system will support a broad range of cuisines rather than focusing on specific types.
- Common dietary preferences will be supported, including vegetarian, vegan, gluten-free, low-sodium, keto, and paleo.

### User Preferences Management
- User preferences will be stored in a simple local dictionary/DataFrame saved as JSON files.
- Preferences will include dietary restrictions, favorite recipes, commonly used ingredients, and past substitutions.
- This approach is chosen for simplicity and ease of implementation within the Kaggle notebook environment.

### Visualization Components
- Recipe images will be displayed where available in the dataset.
- Basic charts will be generated for ingredient proportions and nutritional content.
- Step-by-step instructions will be visually formatted for clarity.

### Balanced Approach
- The system is designed to handle both recipe discovery and cooking guidance equally well.
- The discovery phase helps users find recipes based on available ingredients and preferences.
- The guidance phase provides detailed step-by-step cooking instructions once a recipe is selected.

### Voice Recognition Error Handling
- The system will implement a confirmation step after voice-to-text conversion.
- If recognition errors occur, users can correct via text input or try voice input again.
- Common ingredient name misinterpretations will be handled with a matching algorithm to suggest likely corrections.

