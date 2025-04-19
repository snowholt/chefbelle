Okay, let's craft the content and structure for your Capstone Project showcase website/blog post, designed for a quick build on Wix while meeting all the requirements.

**Project Title Suggestion:** From Fridge Dilemma to Delicious Dinner: An AI-Powered Kitchen Assistant

**Developer Credits:** Nariman Jafarieshlaghi & Neda Saberitabar (Code Enchantresses & AI Integrators)

**Course Context:** Capstone Project for the Google Gen AI Intensive Course (March 31 - April 4, 2025)

---

** Website Structure Suggestion:**

A single-page scrolling website is often fastest to build and keeps the narrative flowing. Use Wix Sections for each part below.

1.  **Hero Section (Top):** Title, Subtitle (brief problem/solution), Developer Names, Course Context, maybe a relevant background image (kitchen, ingredients).
2.  **The Problem Section:** The "What Can I Cook?" Dilemma (using the text you provided).
3.  **Our Solution Section:** Introducing the Interactive Recipe & Kitchen Management Assistant.
4.  **The AI Magic Section:** High-level overview of *key* Gen AI capabilities used. Use icons if possible.
5.  **Under the Hood Section:** Technical deep dive into *each* implemented Gen AI capability with code snippets. This will be the longest section. Use sub-sections or accordions if needed.
6.  **Visual Highlights Section (Optional but Recommended):** Embed the LangGraph visualization, maybe the data exploration charts.
7.  **Real-World Cooking Section:** Limitations & Challenges.
8.  **What's Next? Section:** Future Possibilities.
9.  **About Section:** Brief info about the developers and a link back to the course/Google.

---

**Content for Each Section:**

**(1) Hero Section**

*   **Title:** From Fridge Dilemma to Delicious Dinner: An AI-Powered Kitchen Assistant
*   **Subtitle:** Tackling food waste and mealtime stress with Generative AI. A Capstone Project from the Google Gen AI Intensive Course.
*   **By:** Nariman Jafarieshlaghi & Neda Saberitabar (Code Enchantress & AI Integrators)
*   *(Use a full-width banner, clear text overlays)*

**(2) The Problem: The "What Can I Cook?" Dilemma**

*(Use and slightly expand the text you provided)*

> Home cooks regularly face a “last‑mile” dilemma: they open the fridge or pantry, see a mismatched set of leftover ingredients, and wonder “What can I cook right now without another grocery trip?” This daily challenge often leads to frustration and analysis paralysis.
>
> Traditional recipe apps and websites fall short. They typically:
>
> *   Assume you have *all* listed ingredients.
> *   Lack real‑time awareness of *your* pantry inventory.
> *   Rarely adapt recipes to dietary goals (vegan, low‑sodium, high‑protein) or available substitutions.
> *   Offer limited nutritional insights or user feedback integration.
>
> The consequences? Perfectly good food spoils (over 30% globally!), mealtime becomes stressful, families overspend on take‑out, and culinary creativity is stifled.

*(Use a text block, maybe alongside an image representing the problem - an overflowing fridge or confused cook).*

**(3) Our Solution: The Interactive Kitchen Assistant**

> Introducing our Interactive Recipe & Kitchen Management Assistant – a smart culinary companion designed to bridge the gap between your ingredients and your next meal.
>
> Built as our Capstone project for the Google Gen AI Intensive Course, this assistant leverages the power of Generative AI to help users:
>
> 1.  **Discover recipes** based on ingredients they *actually have*.
> 2.  **Customize recipes** according to dietary needs and preferences.
> 3.  **Receive step-by-step cooking guidance** (future enhancement).
> 4.  **Get nutritional insights** for ingredients and recipes.
> 5.  **Explore user reviews** and ratings for context.
> 6.  **Interact via voice or text**, making it accessible while cooking.

*(Use a text block, perhaps with feature icons or a simple diagram showing input -> AI Assistant -> Output).*

**(4) The AI Magic: How Gen AI Powers the Assistant**

> This isn't just another recipe search engine. We've integrated a suite of cutting-edge Generative AI capabilities to create a truly intelligent kitchen partner:
>
> *   **Understanding You:** Processes **Audio** commands (Speech-to-Text) and natural language text.
> *   **Finding the Perfect Match:** Uses **Embeddings**, **Vector Search (ChromaDB)**, and **Retrieval Augmented Generation (RAG)** to semantically search a vast recipe database based on your ingredients or query.
> *   **Smart Interactions:** Employs **Function Calling** to interact with databases (recipes, reviews) and external APIs (nutrition).
> *   **Orchestrating the Flow:** Leverages **Agents (LangGraph)** to manage the conversation, decide which tools to use, and maintain context (**Stateful Conversation**).
> *   **Factual & Up-to-Date:** Uses **Grounding** (Google Search) for general knowledge and fetches real-time data (**Structured Output/JSON**) from APIs for nutrition.
> *   **Personalized & Evaluated:** Adapts suggestions (future: **Few-Shot Prompting**) and provides **evaluated** responses based on retrieved, structured data, not just hallucinations.

*(Use a multi-column layout with icons and brief text for each capability highlighted).*

**(5) Under the Hood: A Technical Deep Dive**

> Let's explore how these Gen AI capabilities come together. We used Python, Pandas, LangChain/LangGraph, ChromaDB, Google Gemini models, and various APIs.
>
> **a) Data Foundation & RAG (Recipes & Interactions):**
> We started with the Food.com Recipes and User Interactions dataset. After cleaning and preprocessing (handling duplicates, normalizing ingredients, adding basic dietary tags), we created two primary data stores:
> *   **SQLite Database:** For structured querying of recipe details, ingredients, steps, ratings, and reviews.
> *   **ChromaDB Vector Database:** For semantic search. We embedded recipe descriptions/ingredients to find recipes based on meaning, not just keywords.
>
> *How RAG Works Here:* When you ask "What can I make with chicken and broccoli?", the system embeds your query, searches ChromaDB for similar recipe embeddings (Retrieval), and then uses the retrieved recipe information to Generate a relevant answer using the LLM.
>
> *Code Snippet (ChromaDB Setup):*
> ```python
> # Vector Database Setup (ChromaDB)
> def setup_vector_database(
>     vectorized_recipes_df: pd.DataFrame,
>     vectorized_interactions_df: Optional[pd.DataFrame] = None,
>     vector_db_path: str = VECTOR_DB_PATH
> ) -> Tuple[Any, Any, Optional[Any]]:
>     # ... (setup client) ...
>     recipe_collection = client.get_or_create_collection(name="recipes")
>     # ... (process dataframe rows into documents, metadatas, ids) ...
>     for j in range(0, len(recipe_documents), batch_size):
>         end_idx = min(j + batch_size, len(recipe_documents))
>         recipe_collection.add(
>             documents=recipe_documents[j:end_idx],
>             metadatas=recipe_metadatas[j:end_idx],
>             ids=recipe_ids[j:end_idx]
>         )
>     # ... (similar logic for interactions_collection) ...
>     return client, recipe_collection, interactions_collection
> ```
>
> **b) Orchestration with Agents (LangGraph):**
> Managing the conversation flow, deciding when to search the database, call an API, or just chat requires an agent. We used LangGraph to build a stateful agent.
> *   **State (`KitchenState`):** A defined dictionary holding conversation history, user input, current recipe context, tool outputs, etc.
> *   **Nodes:** Python functions representing steps (parsing input, calling tools, processing results, formatting responses).
> *   **Edges:** Logic determining the next node based on the current state (e.g., if the parser detects a request for reviews, route to the review tool).
>
> *Code Snippet (State Definition):*
> ```python
> class KitchenState(TypedDict):
>     messages: Annotated[Sequence[BaseMessage], add_messages]
>     user_input: Optional[str]
>     intent: Optional[str]
>     selected_recipe_id: Optional[str]
>     # ... other state fields for tool outputs, processed data, context ...
>     current_recipe_details: Optional[Dict[str, Any]]
>     recipe_reviews: Optional[Dict[str, Any]]
>     nutritional_info: Optional[Dict[str, Any]]
>     # ... etc. ...
>     finished: bool
>     last_assistant_response: Optional[str]
> ```
> *Code Snippet (Graph Assembly):*
> ```python
> # --- Graph Assembly ---
> graph_builder = StateGraph(KitchenState)
> # Add Nodes
> graph_builder.add_node("InputParserNode", input_parser_node)
> graph_builder.add_node("ToolExecutorNode", tool_executor_node)
> # ... add other custom nodes (AggregateNutrition, ReviewDashboard, etc.)
> graph_builder.add_node("ResponseFormatterNode", response_formatter_node)
> # Define Edges (Entry, Conditional, Standard)
> graph_builder.add_edge(START, "InputParserNode")
> graph_builder.add_conditional_edges("InputParserNode", route_after_parsing, ...)
> graph_builder.add_edge("ToolExecutorNode", "InputParserNode") # Results back to parser
> graph_builder.add_edge("AggregateNutritionNode", "ResponseFormatterNode") # Processed data to formatter
> # ... other edges ...
> graph_builder.add_conditional_edges("ResponseFormatterNode", route_after_formatting, ...)
> # Compile
> kitchen_assistant_graph = graph_builder.compile()
> ```
>
> **c) Function Calling & Structured Output:**
> The agent interacts with the databases and external APIs using predefined tools. The LLM decides *which* tool to call and *what arguments* to pass based on the user's request and the tool descriptions provided in the system prompt. Tools return structured data (JSON strings) that subsequent nodes parse and process.
>
> *Code Snippet (Tool Definition - `get_recipe_by_id`):*
> ```python
> @tool
> def get_recipe_by_id(recipe_id: str) -> str:
>     """
>     Retrieves full details for a specific recipe given its ID from the SQL database.
>     Returns details as a JSON string. Includes 'normalized_ingredients' used for nutrition lookup.
>     """
>     print(f"DEBUG TOOL CALL: get_recipe_by_id(recipe_id='{recipe_id}')")
>     try:
>         # ... (database connection and query logic) ...
>         if not recipe_data:
>             return json.dumps({"status": "not_found", ...})
>         recipe_dict = dict(recipe_data)
>         # ... (parsing logic for list-like fields) ...
>         return json.dumps(recipe_dict, indent=2, default=str)
>     except Exception as e:
>         # ... (error handling) ...
>         return json.dumps({"error": f"Error fetching recipe: {e}"})
> ```
> *Code Snippet (Tool Definition - `fetch_nutrition_from_usda_fdc`):*
> ```python
> @tool
> def fetch_nutrition_from_usda_fdc(ingredient_name: str, api_key: str) -> str:
>     """
>     Fetches nutrition data (per 100g) for a single ingredient from USDA FoodData Central API.
>     Requires a USDA FDC API key. Includes robust retry logic.
>     Returns nutrition data as a JSON string or an error/unavailable status.
>     """
>     print(f"DEBUG TOOL CALL: fetch_nutrition_from_usda_fdc(...)")
>     # ... (API call logic with requests, error handling, retries) ...
>     if data.get('foods'):
>         # ... (extract nutrients using FDC_NUTRIENT_MAP) ...
>         return json.dumps(filtered_nutrition, indent=2)
>     else:
>         return json.dumps({"status": "unavailable", ...})
>     # ... (exception handling) ...
> ```
>
> **d) Audio Understanding:**
> We enabled voice interaction by integrating Speech-to-Text. While Google Cloud STT was initially considered, we opted for OpenAI's Whisper-1 for its simplicity in this demo (though Google STT could be swapped in). The transcribed text is fed into the LangGraph agent as user input.
>
> *Code Snippet (Transcription Function):*
> ```python
> def transcribe_audio(service="openai", file_path=None, language="en", api_key=None, ...):
>     # ... (error checking) ...
>     if service.lower() == "openai":
>         # ... (setup OpenAI client) ...
>         with open(file_path, "rb") as audio_file:
>             transcription = client.audio.transcriptions.create(
>                 model="whisper-1",
>                 file=audio_file,
>                 language=language
>             )
>         return transcription.text
>     elif service.lower() == "google":
>         # ... (setup Google client using credentials) ...
>         # ... (read audio, configure recognition) ...
>         response = client.recognize(config=config, audio=audio)
>         # ... (extract transcript) ...
>     # ... (exception handling) ...
> ```
>
> **e) Grounding:**
> For general cooking questions not covered by the recipe database (like ingredient substitutions), the agent leverages Google Search grounding integrated directly into the Gemini model. This ensures answers are based on broader web knowledge when needed.
>
> *Code Snippet (Invoking LLM with Grounding):*
> ```python
> # LLM Initialization with grounding enabled
> llm = ChatGoogleGenerativeAI(
>     model="gemini-2.0-flash",
>     # ... other params ...
>     generation_config=types.GenerateContentConfig(
>         tools=[types.Tool(google_search=types.GoogleSearch())] # Enable grounding
>     )
> )
> # When the agent decides not to call a specific tool,
> # invoking this LLM automatically uses grounding if relevant.
> ai_response: AIMessage = llm_with_callable_tools.invoke(context_messages)
> ```
>
> **f Context Caching & Stateful Conversation:**
> LangGraph's state management (`KitchenState`) is key. The agent remembers the `selected_recipe_id` and `current_recipe_details` across turns, allowing follow-up questions like "show me the reviews" or "how do I make this vegan?" without the user needing to repeat the recipe ID.

* Use text blocks for explanations and code blocks for snippets. Use subheadings for each capability. Embed the graph PNG here if desired).*

**(6) Visual Highlights (Optional Section)**

*   Embed the LangGraph visualization PNG image here.
*   Optionally, embed one or two of the data exploration charts (like Cuisine Distribution or Ingredients Count).
*(Wix: Use Image blocks and potentially Chart elements if you recreate charts in Wix or embed images of them).*

**(7) Navigating the Real World: Limitations & Challenges**

> Building this AI assistant highlighted both the power and the current limitations of Gen AI:
>
> *   **Data Nuances:** Normalizing ingredient names ("spring onions" vs. "scallions") and accurately tagging recipes (vegan vs. vegetarian with hidden ingredients) remains challenging and requires sophisticated NLP or curated data. Our current tagging is basic.
> *   **API Dependencies:** Relying on external APIs like USDA FDC means potential rate limits, downtime, or changes. We built in retry logic, but it's a factor. (We initially explored Open Food Facts but found USDA FDC more reliable for this use case).
> *   **"Live" Data Brittleness:** The `fetch_live_recipe_data` tool (currently a placeholder) would rely on web scraping, which breaks easily if the source website (e.g., Food.com) changes its layout.
> *   **True Customization:** Our `customize_recipe` tool uses placeholder logic. True AI-driven recipe adaptation (e.g., adjusting baking times when making a recipe gluten-free) is complex and would require more advanced techniques like fine-tuning or sophisticated prompting.
> *   **Context Length:** While LangGraph helps manage state, very long, complex conversations could still exceed the LLM's context window, potentially causing it to "forget" earlier details.
> *   **Evaluation:** Measuring the "success" of a recipe suggestion or customization is subjective and requires user feedback loops, which weren't implemented in this phase.

*(Use a text block, maybe with bullet points).*

**(8) What's Next?: Future Possibilities**

> This Capstone project lays the foundation. Future enhancements could include:
>
> *   **Smart Pantry Integration:** Connect to smart fridge APIs or allow manual inventory tracking for hyper-personalized suggestions.
> *   **Visual Recognition:** Use **Image Understanding** to identify ingredients from user photos.
> *   **Advanced Customization:** Implement **Few-Shot Prompting** or fine-tune a model to generate more reliable recipe modifications.
> *   **Interactive Cooking Mode:** Step-by-step voice/text guidance with timers, technique explanations (potentially **Video Understanding** for linked clips).
> *   **Meal Planning & Shopping Lists:** Generate weekly plans based on inventory, goals, and sales, automatically creating shopping lists.
> *   **Deeper User Profiles:** Store allergies, kitchen equipment, skill level, and taste preferences for even better recommendations.

*(Wix: Use a text block, maybe with icons representing future features).*

**(9) About the Chefs & The Course**

*   **Developers:** Nariman Jafarieshlaghi & Neda Saberitabar (Code Enchantress & AI Integrators)
*   **Project:** Capstone for the Google Gen AI Intensive Course (March 31 - April 4, 2025).
*   *(Optional: Add LinkedIn profile links or a brief bio)*

*(Use a text block, maybe with profile pictures if available).*

---

**Wix Build Tips:**

*   **Template:** Choose a clean, modern template (Portfolio, Tech Blog, Project Showcase). Single-page templates are often good.
*   **Sections:** Use Wix Sections for each content block outlined above. This makes rearranging easy.
*   **Visuals:** Break up text with images (ingredients, kitchen scenes, the graph visualization), icons, and potentially background videos/strips.
*   **Code Snippets:** Use Wix's HTML Embed element or a formatted Text block with a monospace font to display code. Keep snippets short and focused.
*   **Navigation (if multi-page):** Keep it simple: Home, How It Works, Reflections, About.
*   **Mobile Responsiveness:** Check how it looks on mobile using Wix's editor preview.
*   **Publish:** Once you're happy, hit publish!

This structure and content should provide a comprehensive yet concise overview of your project, suitable for showcasing your work effectively on a quickly built Wix site. Good luck!