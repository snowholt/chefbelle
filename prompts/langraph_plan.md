

# LangGraph Implementation Plan for Kitchen Management Assistant (Revised)

## Introduction

This document outlines the revised plan for implementing a LangGraph-based workflow for our Interactive Recipe & Kitchen Management Assistant. LangGraph will orchestrate the various components developed (data storage, search functions, nutrition lookup, audio processing) into a stateful, conversational agent. This plan incorporates insights from the project's progress and the BaristaBot example notebook.

The goal is to create an agent that can:
*   Understand user requests via text or voice.
*   Search for recipes based on various criteria (ingredients, cuisine, dietary needs, time).
*   Retrieve and present detailed recipe information, including ingredients, steps, ratings, reviews, and nutritional data.
*   Customize recipes using few-shot prompting.
*   Answer general cooking questions, potentially using web grounding.
*   Manage conversation state and user preferences.

## 1. State Schema Definition (`KitchenState`)

### Title: Define the Core State Schema

#### Description
Define the `TypedDict` schema for the assistant's state. This schema will hold all necessary information passed between graph nodes.

#### Task Details
- Create a `KitchenState` TypedDict with clear annotations.
- Include conversation history (`messages`) using `add_messages`.
- Define slots for user input (text/audio), search parameters, search results, selected recipe details, customization requests, nutritional information, and grounding results.
- Add fields for user context like available ingredients and dietary preferences.
- Include flags for control flow (e.g., `needs_clarification`, `finished`).

#### Implementation Example
```python
from typing import Annotated, List, Dict, Optional, Any, Sequence
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class KitchenState(TypedDict):
    """State representing the kitchen assistant conversation."""
    # Conversation history (Human, AI, Tool messages)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # User's raw input (text or transcribed audio)
    user_input: Optional[str]

    # Parsed intent from user input
    intent: Optional[str] # e.g., 'search_recipe', 'get_details', 'customize', 'nutrition', 'general_chat', 'grounding_query'

    # Parameters extracted for specific actions
    search_params: Dict[str, Any] # {'query_text', 'cuisine', 'dietary_tag', 'max_minutes'}
    selected_recipe_id: Optional[str]
    customization_request: Optional[str]
    nutrition_query: Optional[str] # Ingredient or recipe name
    grounding_query: Optional[str] # Question for web search

    # Data retrieved by tools/nodes
    search_results: Optional[List[Dict[str, Any]]] # List of recipe summaries from search
    current_recipe_details: Optional[Dict[str, Any]] # Full details of selected recipe
    recipe_reviews: Optional[Dict[str, Any]] # Ratings and reviews
    nutritional_info: Optional[Dict[str, Any]] # Fetched nutrition data
    grounding_results: Optional[str] # Results from web search

    # User Context
    user_ingredients: List[str] # Ingredients the user has
    dietary_preferences: List[str] # e.g., ['vegetarian', 'gluten-free']

    # Control Flow
    needs_clarification: bool # Flag if agent needs more info from user
    finished: bool # Flag indicating end of conversation
    last_assistant_response: Optional[str] # Store the last text response for UI display
```

## 2. System Instructions & Core Nodes

### Title: Define System Instructions and Core Nodes

#### Description
Establish the guiding system prompt for the Gemini model and implement the fundamental nodes for managing the conversation flow and user interaction.

#### Task Details
- Define detailed system instructions (`KITCHEN_ASSISTANT_SYSINT`) outlining the assistant's capabilities (recipe search, details, customization, nutrition, voice, grounding), personality, and rules for tool usage (especially specifying `limit` for reviews).
- Implement an `InputParserNode`: Processes the latest user message (from text or transcribed audio), uses the LLM (Gemini) to determine user intent, extracts relevant parameters (recipe ID, search terms, customization details), and updates the `intent` and parameter fields in the state. This node also handles general chat responses if no specific tool/action is identified.
- Implement a `HumanInputNode`: Handles interaction with the user. For text, it prompts for input. For voice, it could trigger recording/transcription (or receive transcribed text). It updates `user_input` and potentially the `finished` flag (if user quits).
- Implement a `ResponseFormatterNode`: Takes the results from various action nodes (search results, recipe details, etc.) and formats them into a user-friendly natural language response, updating `last_assistant_response`.

#### Implementation Notes
- The `InputParserNode` is crucial for understanding the user and deciding the next step. It will leverage the LLM's NLU capabilities.
- The `HumanInputNode` will be adapted based on the input modality (text vs. voice).

## 3. Tool Definition and Integration

### Title: Define and Integrate Tools

#### Description
Define the Python functions developed in `capstone-2025-kma-nn.ipynb` as tools that the LangGraph agent can invoke. Differentiate between stateless tools (executed by `ToolNode`) and stateful actions (handled by custom nodes).

#### Task Details
- **Stateless Tools (for `ToolNode`):**
    - `gemini_recipe_similarity_search`: Performs vector search for recipes.
    - `gemini_interaction_similarity_search`: Performs vector search for reviews.
    - `get_recipe_by_id`: Fetches structured recipe data from SQLite (excluding live nutrition).
    - `get_ratings_and_reviews_by_recipe_id`: Fetches ratings/reviews from SQLite. **Ensure the LLM is instructed to always provide the `limit` parameter.**
    - `fetch_nutrition_from_openfoodfacts`: Fetches nutrition for a *single* ingredient (can be used directly or within a custom nutrition node).
    - `google_search`: (Built-in or custom wrapper) For web grounding.
- **Stateful Actions (handled by custom nodes):**
    - *Recipe Customization*: Requires LLM reasoning with few-shot examples, handled in `RecipeCustomizationNode`.
    - *Nutrition Aggregation*: Summing nutrition across ingredients, handled in `NutritionAnalysisNode`.
    - *User Preference/Ingredient Updates*: Modifying `dietary_preferences` or `user_ingredients` in the state, potentially handled in an `UpdateStateNode` or within the `InputParserNode`.
- Bind the appropriate tools to the LLM within the `InputParserNode` so it knows when to call them.
- Implement a `ToolExecutorNode`: Use LangGraph's `ToolNode` to execute the stateless tools identified by the `InputParserNode`.

## 4. Specific Action Nodes

### Title: Implement Nodes for Core Functionalities

#### Description
Create dedicated nodes for each primary capability of the assistant, leveraging the defined tools and state.

#### Task Details
- **`RecipeSearchNode`**:
    - Triggered when `intent` is `search_recipe`.
    - Takes `search_params` from the state.
    - Calls the `gemini_recipe_similarity_search` tool via the `ToolExecutorNode`.
    - Stores results in `search_results`.
    - Transitions to `ResponseFormatterNode`.
- **`RecipeDetailNode`**:
    - Triggered when `intent` is `get_details`.
    - Takes `selected_recipe_id` from the state.
    - Calls `get_recipe_by_id` and `get_ratings_and_reviews_by_recipe_id` tools via `ToolExecutorNode`.
    - Stores results in `current_recipe_details` and `recipe_reviews`.
    - Transitions to `NutritionAnalysisNode` (to fetch ingredient nutrition) or directly to `ResponseFormatterNode`.
- **`NutritionAnalysisNode`**:
    - Triggered when `intent` is `nutrition` or after `RecipeDetailNode`.
    - Takes `nutrition_query` (ingredient name) or `current_recipe_details` (for recipe analysis) from the state.
    - If analyzing a recipe, iterates through ingredients in `current_recipe_details`, calls `fetch_nutrition_from_openfoodfacts` tool for each (via `ToolExecutorNode`), aggregates results, and stores in `nutritional_info`.
    - If analyzing a single ingredient, calls the tool directly.
    - Transitions to `ResponseFormatterNode`.
- **`RecipeCustomizationNode`**:
    - Triggered when `intent` is `customize`.
    - Takes `current_recipe_details` and `customization_request` from the state.
    - Uses the LLM (Gemini) with **Few-Shot Prompting**: Provide examples of successful customizations (vegetarian, gluten-free, lower-calorie substitutions) in the prompt.
    - Generates the modified recipe text/steps.
    - Stores the result potentially back in `current_recipe_details` (as a modified version) or a separate state field.
    - Transitions to `ResponseFormatterNode`.
- **`AudioInputNode`**:
    - Triggered when voice input is received.
    - Takes the audio file path/data from the state (or external trigger).
    - Calls the `transcribe_audio` function (likely outside the main graph loop or as the entry point for voice).
    - Updates `user_input` with the transcribed text.
    - Transitions to `InputParserNode`.
- **`WebGroundingNode`**:
    - Triggered when `intent` is `grounding_query` or when the `InputParserNode` determines external info is needed.
    - Takes `grounding_query` from the state.
    - Calls the `google_search` tool via the `ToolExecutorNode`.
    - Stores results in `grounding_results`.
    - Transitions to `ResponseFormatterNode`.

## 5. Conditional Edge Functions

### Title: Define Conditional Transitions

#### Description
Create functions to dynamically route the workflow based on the current state and the output of the `InputParserNode`.

#### Task Details
- Implement `route_after_parsing`: Takes the state (specifically the `intent` field set by `InputParserNode`) and returns the name of the next node (e.g., "RecipeSearchNode", "RecipeDetailNode", "HumanInputNode", "WebGroundingNode", "END").
- Implement `route_after_human`: Checks the `finished` flag in the state. If true, returns `END`; otherwise, returns "InputParserNode".
- Implement `route_after_tool_execution`: Determines the next step after a tool runs (usually back to `InputParserNode` to process results, or potentially to `ResponseFormatterNode`).

#### Implementation Example (`route_after_parsing`)
```python
def route_after_parsing(state: KitchenState) -> str:
    """Route based on the intent determined by the InputParserNode."""
    intent = state.get("intent")
    if intent == "search_recipe":
        return "RecipeSearchNode"
    elif intent == "get_details":
        return "RecipeDetailNode"
    elif intent == "customize":
        return "RecipeCustomizationNode"
    elif intent == "nutrition":
        return "NutritionAnalysisNode"
    elif intent == "grounding_query":
        return "WebGroundingNode"
    elif state.get("needs_clarification"):
        return "HumanInputNode" # Ask user for more info
    elif intent == "exit":
         return END
    else: # Default or general chat
        return "ResponseFormatterNode" # Format the chat response
```

## 6. Graph Assembly and Compilation

### Title: Assemble and Compile the Complete Graph

#### Description
Define the full graph structure connecting all nodes and edges, then compile it into an executable LangGraph application.

#### Task Details
- Instantiate `StateGraph(KitchenState)`.
- Add all defined nodes (`InputParserNode`, `HumanInputNode`, `ToolExecutorNode`, `RecipeSearchNode`, `RecipeDetailNode`, `NutritionAnalysisNode`, `RecipeCustomizationNode`, `WebGroundingNode`, `ResponseFormatterNode`, potentially `AudioInputNode` if integrated differently).
- Define the entry point (likely `InputParserNode` or potentially `AudioInputNode` depending on UI).
- Add direct edges (e.g., `ToolExecutorNode` -> `InputParserNode`, `ResponseFormatterNode` -> `HumanInputNode`).
- Add conditional edges using the routing functions (e.g., from `InputParserNode` based on `intent`, from `HumanInputNode` based on `finished`).
- Compile the graph using `graph_builder.compile()`.

## 7. User Interface Integration (Conceptual)

### Title: Integrate with User Interface

#### Description
Connect the compiled LangGraph application with the conceptual user interface components (text input, voice selection/recording).

#### Task Details
- **Input**:
    - Text: Pass text from the UI input field to the `HumanInputNode`.
    - Voice: Trigger `AudioInputNode` with the selected/recorded audio file path. The transcribed text then flows to `InputParserNode`.
- **Output**:
    - Display `last_assistant_response` from the state in the UI chat area.
    - Format recipe search results, details, nutritional info, etc., appropriately for display.
- **State**: Maintain conversation history (`messages`) and potentially other relevant state parts (like current search results) for display in the UI.

## 8. Testing and Refinement

### Title: Test and Refine the Complete System

#### Description
Conduct thorough testing covering various user scenarios, interaction flows, and edge cases. Refine prompts, routing logic, and node implementations based on test results.

#### Task Details
- **Scenario Testing**:
    - Simple recipe search ("Find chicken recipes").
    - Filtered search ("Find vegan Italian pasta under 30 minutes").
    - Recipe details request ("Tell me more about recipe 12345").
    - Nutrition query ("What's the nutrition for butter?", "How many calories in recipe 12345?").
    - Customization ("Make recipe 12345 gluten-free", "Substitute tofu for chicken in recipe 67890").
    - Grounding ("What's a good substitute for eggs in baking?").
    - Voice commands for all above scenarios.
    - Multi-turn conversations involving clarification and refinement.
    - Error handling (invalid recipe ID, API failures, unclear requests).
- **Validation**: Check if the correct nodes are activated, tools are called with correct parameters, state is updated appropriately, and responses are accurate and helpful.
- **Refinement**: Adjust system prompts, few-shot examples, routing conditions, and tool descriptions based on performance.

## Conclusion

This revised plan provides a detailed roadmap for building the Kitchen Management Assistant using LangGraph. It leverages the completed data processing, storage, and function-calling components, integrates the planned Gen AI capabilities (RAG, Function Calling, Few-Shot, Grounding, Audio), and follows patterns from the BaristaBot example for a robust, stateful agent architecture. The focus is now on implementing the nodes, edges, and routing logic within the LangGraph framework.

---