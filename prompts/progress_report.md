# Interactive Recipe & Kitchen Management Assistant - Progress Report (April 12, 2025)

## Project Goal
Develop an AI-powered assistant using Google Gemini and LangGraph to help users discover, customize, and manage recipes through natural language interaction, including voice commands.

---

#### Step 1: Data Acquisition, Cleaning, and Storage
**Status**: ✅ Completed

**Summary**:
- Acquired and loaded the Food.com recipe and interaction datasets (`RAW_recipes.csv`, `RAW_interactions.csv`) and a nutrition dataset (`cleaned_nutrition_dataset.csv`).
- Performed data exploration and identified key features.
- Cleaned the data by removing duplicates (`check_remove_duplicates`) and normalizing ingredient names (`normalize_ingredients`).
- Implemented basic dietary tag identification based on ingredients (`identify_dietary_tags`).
- Established a hybrid storage system:
    - **SQLite Database (`kitchen_db.sqlite`)**: Stores structured recipe, interaction, and nutrition data (`setup_sql_database`).
    - **ChromaDB Vector Store (`vector_db/`)**: Stores recipe and interaction embeddings for semantic search (`setup_vector_database`).

**Gen AI Capabilities Demonstrated**:
- Data preparation for document understanding and RAG.

**Challenges & Solutions**:
- Handled large dataset loading and processing efficiently.
- Addressed column naming inconsistencies and string list parsing.
- Optimized performance by replacing heavy analytics with targeted ones.
- Created a robust hybrid storage solution linking structured and vector data.

---

#### Step 2: Audio Input & Command Recognition
**Status**: 🔄 Partially Completed

**Summary**:
- Implemented a function (`transcribe_audio`) for speech-to-text conversion supporting OpenAI Whisper-1 and Google Cloud Speech-to-Text (with Whisper-1 preferred due to simpler setup).
- Set up basic infrastructure for handling audio files (OGG, WAV).
- Initial setup for a dual text/voice interface concept. Full integration and user preference management are pending.

**Gen AI Capabilities Demonstrated**:
- Audio understanding (Speech-to-Text).
- Multimodal input handling concept.

**Challenges & Solutions**:
- Addressed complexities in Google Cloud Speech-to-Text authentication by favoring Whisper-1.
- Handled different audio formats.

---

#### Step 3: Recipe Search & Details Retrieval (Function Calling Foundation)
**Status**: ✅ Completed

**Summary**:
- Developed functions for interacting with the hybrid storage system:
    - SQL Database: `list_tables`, `describe_table`, `execute_query`.
    - Recipe Retrieval: `get_recipe_by_id`, `get_ratings_and_reviews_by_recipe_id`.
- Integrated live nutritional data fetching for ingredients using the Open Food Facts API (`fetch_nutrition_from_openfoodfacts`).
- Configured the Gemini model (`gemini-2.0-flash`) to use these functions as tools, enabling natural language queries for recipe details and reviews.

**Key Technologies Used**:
- SQLite, ChromaDB, Google Gemini API (Function Calling), Open Food Facts API, Pandas.

**Gen AI Capabilities Demonstrated**:
- Function Calling (Tool Use).
- Natural Language Understanding for database interaction.
- Structured Output Generation (API responses).

**Challenges & Solutions**:
- Designed effective function schemas for Gemini.
- Ensured robust error handling for external API calls (Open Food Facts).
- Handled parsing of potentially inconsistent data formats stored in the database.

---

#### Step 4: RAG Implementation for Semantic Search
**Status**: ✅ Completed

**Summary**:
- Implemented Retrieval Augmented Generation (RAG) using the ChromaDB vector store.
- Developed semantic search functions (`gemini_recipe_similarity_search`, `gemini_interaction_similarity_search`) to find recipes or reviews based on natural language queries and similarity.
- Added metadata filtering capabilities (cuisine, dietary tags, cooking time) to the semantic search.

**Key Technologies Used**:
- ChromaDB, Google Gemini API, Pandas.

**Gen AI Capabilities Demonstrated**:
- Retrieval Augmented Generation (RAG).
- Semantic Search (Vector Embeddings).
- Natural Language Understanding for complex queries.

**Challenges & Solutions**:
- Optimized ChromaDB setup for efficient batch processing.
- Constructed meaningful document representations for effective vector search.
- Integrated RAG search with metadata filtering.

---


#### Step 5: Grounding with Google Search
**Status**: ✅ Demonstrated

**Summary**:
- Demonstrated the use of Google Search grounding with the Gemini API (`google_search` tool).
- Showcased how grounding can be used to fetch up-to-date external information to answer questions beyond the internal dataset (e.g., finding ingredient substitutes). Integration into the main agent is pending.

**Key Technologies Used**:
- Google Gemini API (Grounding).

**Gen AI Capabilities Demonstrated**:
- Grounding.

**Challenges & Solutions**:
- Simple demonstration; full integration requires deciding when grounding is necessary within the agent flow.

---

#### Step 5: Function Calling & AI Agent with LangGraph
**Status**: 🔄 In Progress

**Summary**:
- Implemented Step 1 of the LangGraph plan: Defined the KitchenState TypedDict schema. This schema includes fields for conversation history, user input, parsed intent, action parameters, retrieved data (search results, recipe details, nutrition, grounding), user context (ingredients, preferences), and control flow flags. It utilizes Annotated and add_messages for proper message handling within LangGraph.
- Defined KITCHEN_ASSISTANT_SYSINT with detailed instructions for the Gemini model, covering capabilities and tool usage rules (including the limit parameter for reviews).
- Created input_parser_node: Simulates LLM-based intent recognition and parameter extraction from user input (text/transcribed audio). Handles basic chat responses and flags need for clarification. (Full LLM integration pending).
- Created human_input_node: Manages text-based user interaction, displays the last assistant response, gets user input, and detects exit commands.
- Created response_formatter_node: Takes processed data from the state (search results, recipe details, nutrition info, grounding results) and formats it into a user-friendly natural language response, updating last_assistant_response. 
- Clears processed data fields from the state.
- Defined stateless tools (gemini_recipe_similarity_search, gemini_interaction_similarity_search, get_recipe_by_id, get_ratings_and_reviews_by_recipe_id, fetch_nutrition_from_openfoodfacts, google_search) using the @tool decorator, based on functions from the capstone notebook.
- Created placeholders using @tool for stateful actions (customize_recipe, calculate_recipe_nutrition, update_user_context) to allow the LLM to recognize these intents.
- Instantiated LangGraph's ToolNode (tool_executor_node) with the list of stateless tools for automatic execution.
- Updated the input_parser_node to use ChatGoogleGenerativeAI bound with all defined tools (llm_with_all_tools). This enables the LLM to generate tool_calls based on user input and system instructions. The node now processes the LLM's - AIMessage to either extract tool calls or handle direct chat responses/clarifications, updating the state's intent accordingly.
**Key Technologies Used**:
- LangGraph, Google Gemini API, SQLite, ChromaDB, Open Food Facts API.

**Gen AI Capabilities Being Implemented**:
- Agent-based reasoning and planning.
- Advanced Function Calling / Tool Use orchestration.
- State management and context maintenance.

**Challenges & Solutions**:
- Designing a flexible and robust agent graph.
- Ensuring seamless tool integration and error handling within the agent flow.
- Managing conversational state effectively.

---





#### Step 7: User Interface, Testing, and Deployment
**Status**: ⬜ Not Started

**Summary**:
- Development of a user-friendly interface, comprehensive testing, and deployment strategy will follow the completion of the core agent logic.

---

## Overall Progress Summary

**Completed**: Data processing, hybrid storage setup, core function calling tools (DB access, RAG search, nutrition lookup), RAG implementation, grounding demonstration.
**In Progress**: LangGraph agent development (Step 5), full voice integration (Step 2).
**Not Started**: UI, Testing, Deployment (Step 7).

### Current Focus
- Building and refining the LangGraph agent to orchestrate the various tools and capabilities for complex recipe management tasks.

### Next Steps
- Complete the LangGraph agent implementation (Step 5).
- Fully integrate voice input/output (Step 2).
- Begin planning for UI and testing (Step 7).