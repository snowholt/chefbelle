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
- Implemented a function (`transcribe_audio`) for speech-to-text conversion supporting OpenAI Whisper-1 (preferred) and Google Cloud Speech-to-Text.
- Set up basic infrastructure for handling audio files (OGG, WAV).
- Conceptual UI integration includes a placeholder for voice file selection and triggering transcription.
- Full integration into the LangGraph flow (e.g., dedicated `AudioInputNode`) and robust user preference management are pending.

**Gen AI Capabilities Demonstrated**:
- Audio understanding (Speech-to-Text).
- Multimodal input handling concept.

**Challenges & Solutions**:
- Simplified setup by favoring Whisper-1 over Google STT authentication complexities.
- Need to design the trigger mechanism for voice input within the agent/UI loop.

---

#### Step 3: Recipe Search & Details Retrieval (Function Calling Foundation)
**Status**: ✅ Completed

**Summary**:
- Developed Python functions for interacting with the hybrid storage system:
    - SQL Database: `list_tables`, `describe_table`, `execute_query`.
    - Recipe Retrieval: `get_recipe_by_id`, `get_ratings_and_reviews_by_recipe_id`.
- Integrated live nutritional data fetching for *individual* ingredients using the Open Food Facts API (`fetch_nutrition_from_openfoodfacts`).
- These functions are now defined as tools for the LangGraph agent.

**Key Technologies Used**:
- SQLite, ChromaDB, Open Food Facts API, Python (`@tool` decorator).

**Gen AI Capabilities Demonstrated**:
- Function definition for Tool Use.
- Database interaction logic.
- External API integration.

**Challenges & Solutions**:
- Designed functions to return structured data (JSON strings) suitable for agent processing.
- Ensured robust error handling for external API calls.
- Handled parsing of potentially inconsistent data formats stored in the database within retrieval functions.

---

#### Step 4: RAG Implementation for Semantic Search
**Status**: ✅ Completed

**Summary**:
- Implemented Retrieval Augmented Generation (RAG) using the ChromaDB vector store.
- Developed semantic search functions (`gemini_recipe_similarity_search`, `gemini_interaction_similarity_search`) now defined as tools for the agent.
- Added metadata filtering capabilities (cuisine, dietary tags, cooking time) to the `gemini_recipe_similarity_search` tool.

**Key Technologies Used**:
- ChromaDB, Python (`@tool` decorator).

**Gen AI Capabilities Demonstrated**:
- Retrieval Augmented Generation (RAG) function definition.
- Semantic Search (Vector Embeddings) function definition.
- Metadata filtering within search tools.

**Challenges & Solutions**:
- Optimized ChromaDB setup for efficient batch processing during initial setup.
- Constructed meaningful document representations for effective vector search during initial setup.
- Encapsulated search and filtering logic into reusable tool functions.

---

#### Step 5: Grounding with Google Search
**Status**: ✅ Demonstrated & Tool Defined

**Summary**:
- Demonstrated the use of Google Search grounding with the Gemini API directly.
- Defined a placeholder `google_search` tool using the `@tool` decorator for integration into the LangGraph agent. Full API integration within the tool is pending but the agent can now *call* it.

**Key Technologies Used**:
- Google Gemini API (Grounding concept), Python (`@tool` decorator).

**Gen AI Capabilities Demonstrated**:
- Grounding (concept and tool definition).

**Challenges & Solutions**:
- Tool defined, but requires actual Google Search API implementation/credentials for full functionality within the agent.

---

#### Step 6: LangGraph Agent Implementation
**Status**: ✅ Partially Completed (Core Structure Built & Tested)

**Summary**:
- **State Schema (`KitchenState`)**: Defined a comprehensive `TypedDict` schema to manage conversation history, user input, intents, parameters, tool results, user context, and control flow flags.
- **System Instructions**: Created detailed system prompts (`KITCHEN_ASSISTANT_SYSINT`) guiding the LLM's behavior, capabilities, and tool usage rules (including `limit` for reviews).
- **Core Nodes**:
    - `InputParserNode`: Implemented using `ChatGoogleGenerativeAI` bound with all defined tools. Parses user input, determines intent, extracts parameters (via tool call arguments), and generates `AIMessage` (with content or `tool_calls`).
    - `ResponseFormatterNode`: Implemented to format tool results or direct LLM responses into user-friendly text, updating `last_assistant_response`.
    - `HumanInputNode` (Conceptual): Logic handled outside the graph via `ipywidgets` for notebook interaction.
- **Tool Integration**:
    - Defined stateless tools (`gemini_recipe_similarity_search`, `get_recipe_by_id`, `get_ratings_and_reviews_by_recipe_id`, `fetch_nutrition_from_openfoodfacts`, `google_search`) using `@tool`.
    - Defined placeholder tools (`customize_recipe`, etc.) for intent recognition.
    - Implemented `ToolExecutorNode` using `langgraph.prebuilt.ToolNode` to execute stateless tools.
- **Action Nodes (Placeholders/Preparation)**:
    - Implemented nodes (`recipe_search_node`, `recipe_detail_node`, `nutrition_analysis_node`, `web_grounding_node`) primarily to *prepare* the `AIMessage` with the correct `tool_calls` for the `ToolExecutorNode` based on the parser's intent.
    - `RecipeCustomizationNode` implemented as a placeholder, indicating where few-shot LLM logic will reside.
    - `NutritionAnalysisNode` prepares calls for `fetch_nutrition_from_openfoodfacts`; aggregation logic is deferred.
- **Routing**: Implemented conditional edge functions (`route_after_parsing`, `route_after_human_or_audio`) to direct the flow between parsing, tool execution, customization, response formatting, and ending the conversation.
- **Graph Assembly**: Assembled and compiled the nodes and edges into an executable `StateGraph` (`kitchen_assistant_graph`).
- **Conceptual UI & Testing**:
    - Implemented a basic UI simulation using `ipywidgets` to interact with the graph via `.stream()`.
    - Set up a testing framework using `test_scenario` function and `.invoke()` to run predefined test cases and validate flow/outputs.

**Key Technologies Used**:
- LangGraph, Google Gemini API (Function Calling), Python (`TypedDict`, `@tool`).

**Gen AI Capabilities Demonstrated/Implemented**:
- Agent-based reasoning (structure defined).
- Advanced Function Calling / Tool Use orchestration (graph defined).
- State management and context maintenance (`KitchenState`).
- Natural Language Understanding (within `InputParserNode`).

**Challenges & Solutions**:
- Designed a state schema to hold diverse information.
- Structured the graph flow with conditional logic for different intents.
- Differentiated between stateless tool execution (`ToolNode`) and custom node logic (placeholders).
- Created a functional testing loop within the notebook using `ipywidgets` and `.stream()`.
- Set up a scenario-based testing framework using `.invoke()`.

---

#### Step 7: User Interface, Full Testing, and Deployment
**Status**: ⬜ Not Started (Conceptual UI in place)

**Summary**:
- A conceptual UI using `ipywidgets` exists for testing within the notebook.
- Development of a dedicated user interface, comprehensive end-to-end testing (including refining prompts, few-shot examples, and node logic), and deployment strategy are future steps.

---

## Overall Progress Summary

**Completed**:
- Data Processing & Storage (SQLite, ChromaDB)
- Core Function/Tool Definitions (DB Access, RAG Search, Nutrition Lookup, Grounding Placeholder, Audio Placeholder)
- RAG Implementation (via tools)
- LangGraph Core Structure (State, Nodes, Edges, Compilation)
- Conceptual UI & Testing Framework for LangGraph

**In Progress**:
- Full implementation of complex LangGraph nodes (e.g., `RecipeCustomizationNode` with few-shot, `NutritionAnalysisNode` aggregation).
- Integration of actual audio input/transcription into the graph flow.
- Integration of actual Google Search API into the `google_search` tool.
- Refinement of LLM prompts and routing logic based on testing.

**Not Started**:
- Dedicated User Interface development.
- Comprehensive end-to-end testing and evaluation.
- Deployment strategy.

### Current Focus
- Refining the existing LangGraph agent based on initial tests.
- Implementing the detailed logic within placeholder nodes (Customization, Nutrition Aggregation).
- Integrating functional audio and web search tools.

### Next Steps
- Implement full logic for `RecipeCustomizationNode` (using few-shot prompting) and `NutritionAnalysisNode` (aggregation).
- Replace placeholder `google_search` and `transcribe_audio` tools/functions with actual implementations.
- Integrate the `AudioInputNode` properly into the graph start or UI loop.
- Conduct extensive testing using the `test_scenario` framework and refine prompts/logic.
- Plan and begin development of a more robust User Interface.