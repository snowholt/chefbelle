
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