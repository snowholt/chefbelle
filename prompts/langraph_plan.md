# LangGraph Implementation Plan for Kitchen Management Assistant

## Introduction

This document outlines the plan for implementing a LangGraph-based workflow for our Interactive Recipe & Kitchen Management Assistant. LangGraph will allow us to create a stateful, graph-based application that effectively leverages the various components we've already built.

## 1. State Schema Definition

### Title: Define the Core State Schema

#### Description
Define the TypedDict schema for our assistant's state, which will be passed between different nodes in the graph.

#### Task Details
- Create a `KitchenState` TypedDict with appropriate annotations
- Include conversation history tracking
- Define slots for current recipe search parameters
- Add storage for recipe customization requests
- Include user preference tracking

#### Implementation Example
```python
from typing import Annotated, List, Dict, Optional, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class KitchenState(TypedDict):
    """State representing the kitchen assistant conversation."""
    # The chat conversation history
    messages: Annotated[list, add_messages]
    
    # Search parameters for finding recipes
    search_params: Dict[str, Any]
    
    # Current recipe under consideration
    current_recipe: Optional[Dict[str, Any]]
    
    # User's dietary preferences
    dietary_preferences: List[str]
    
    # Flag indicating end of conversation
    finished: bool
```

## 2. System Instructions & Core Nodes

### Title: Define System Instructions and Core Nodes

#### Description
Create system instructions for the assistant and implement the basic nodes for the conversation flow.

#### Task Details
- Define detailed system instructions that guide the assistant's behavior
- Implement a chatbot node that processes messages using Gemini
- Create a human input node to handle user interactions
- Add a routing node to direct workflow based on identified intents

#### Implementation Example
```python
KITCHEN_ASSISTANT_SYSINT = (
    "system",
    "You are a Kitchen Management Assistant that helps users discover recipes, customize them according to dietary needs, and provide cooking guidance. You can search for recipes by ingredients, cuisine types, or dietary restrictions. You can also provide nutritional information and suggest substitutions for ingredients..."
)

def chatbot_node(state: KitchenState) -> KitchenState:
    """Process the current conversation state using the Gemini model."""
    # Implementation details here...

def human_node(state: KitchenState) -> KitchenState:
    """Handle user input and update the state accordingly."""
    # Implementation details here...
```

## 3. Recipe Search Node

### Title: Implement Recipe Search Functionality

#### Description
Create a node to handle recipe search requests using our existing vector database functions.

#### Task Details
- Implement intent recognition for recipe search queries
- Extract search parameters from user messages
- Use the `gemini_recipe_similarity_search` function to find matching recipes
- Format and return search results

#### Few-Shot Prompting Example
```
User: "I'm looking for a quick vegetarian pasta recipe"
Assistant: Let me search for that. What specific ingredients would you like to include?
User: "I have tomatoes, garlic, and basil"
Assistant: Great! Let me find some vegetarian pasta recipes with those ingredients.
[SEARCH EXECUTION]
I found several vegetarian pasta recipes that use tomatoes, garlic and basil:
1. Simple Tomato Basil Pasta (15 minutes)
2. Garlic Tomato Linguine (20 minutes)
3. Mediterranean Pasta Primavera (25 minutes)
Would you like details about any of these recipes?
```

## 4. Recipe Detail Node

### Title: Create Recipe Detail Retrieval Node

#### Description
Implement a node to fetch and present detailed information about specific recipes.

#### Task Details
- Create a node that triggers on recipe detail requests
- Use `get_recipe_by_id` function to retrieve complete recipe information
- Format and present recipes with ingredients, steps, and nutritional information
- Include ratings and reviews using `get_ratings_and_reviews_by_recipe_id`

#### Implementation Example
```python
def recipe_detail_node(state: KitchenState) -> KitchenState:
    """Fetch and present detailed recipe information."""
    # Extract recipe ID from state
    recipe_id = extract_recipe_id(state)
    
    # Get recipe details
    recipe = get_recipe_by_id(recipe_id)
    
    # Get ratings and reviews
    reviews = get_ratings_and_reviews_by_recipe_id(recipe_id, limit=3)
    
    # Format response
    response = format_recipe_detail(recipe, reviews)
    
    return state | {"messages": [("assistant", response)]}
```

## 5. Recipe Customization Node with Few-Shot Learning

### Title: Implement Recipe Customization with Few-Shot Learning

#### Description
Create a node to handle recipe customization requests using few-shot prompting techniques.

#### Task Details
- Implement few-shot prompting for common recipe modifications
- Handle dietary restriction adaptations
- Support ingredient substitutions based on user preferences
- Adjust cooking times and serving sizes

#### Few-Shot Examples to Include
```
Example 1:
User: "Can you make this recipe vegetarian?"
Assistant: [FEW-SHOT RESPONSE: Shows how to replace meat with plant-based alternatives while maintaining flavor profile]

Example 2:
User: "I need to make this gluten-free."
Assistant: [FEW-SHOT RESPONSE: Shows how to substitute wheat-based ingredients with gluten-free alternatives]

Example 3:
User: "I want to reduce the calories in this recipe."
Assistant: [FEW-SHOT RESPONSE: Shows how to modify cooking techniques and ingredients to reduce calorie content]
```

## 6. Nutrition Analysis Node

### Title: Create Nutrition Analysis Node

#### Description
Implement a node that can analyze the nutritional content of recipes and ingredients.

#### Task Details
- Create a node that triggers on nutrition-related queries
- Use `fetch_nutrition_from_openfoodfacts` to get nutritional data for ingredients
- Implement aggregation of nutritional values across recipe ingredients
- Format and present nutritional information with comparisons to daily values

#### Implementation Example
```python
def nutrition_node(state: KitchenState) -> KitchenState:
    """Analyze and present nutritional information for recipes or ingredients."""
    # Implementation details here...
```

## 7. Audio Input Handling

### Title: Implement Voice Input Processing

#### Description
Create a node to handle voice input using our existing transcription functionality.

#### Task Details
- Implement a node that processes audio input
- Use the `transcribe_audio` function to convert speech to text
- Update the conversation state with the transcribed text
- Ensure seamless transition between voice and text interactions

#### Implementation Example
```python
def voice_input_node(state: KitchenState) -> KitchenState:
    """Process voice input and update the conversation state."""
    # Implementation details here...
```

## 8. Web Grounding Node

### Title: Add Web Search Grounding

#### Description
Create a node that can supplement recipe information with data from the web.

#### Task Details
- Implement a node that triggers when additional information is needed
- Use Google Search API to find relevant supplemental information
- Integrate web search results into recipe recommendations and answers
- Present grounded information with proper attribution

#### Implementation Example
```python
def web_grounding_node(state: KitchenState) -> KitchenState:
    """Supplement assistant responses with web search results."""
    # Implementation details here...
```

## 9. Conditional Edge Functions

### Title: Define Conditional Transitions

#### Description
Create functions to determine the next node in the graph based on the current state.

#### Task Details
- Implement intent recognition to route between nodes
- Define exit conditions for ending the conversation
- Ensure appropriate handling of context across multiple turns
- Create specialized routing for voice vs. text input

#### Implementation Example
```python
def route_based_on_intent(state: KitchenState) -> str:
    """Route to the appropriate node based on detected intent."""
    # Extract the last user message
    last_msg = state["messages"][-1]
    
    # Detect intent (simplified example)
    if "search" in last_msg.lower():
        return "recipe_search"
    elif "nutrition" in last_msg.lower():
        return "nutrition_analysis"
    # More routing conditions...
    
    # Default to general chat
    return "chatbot"
```

## 10. Graph Assembly and Compilation

### Title: Assemble and Compile the Complete Graph

#### Description
Define the full graph structure with all nodes and transitions, then compile it for execution.

#### Task Details
- Add all nodes to the graph
- Define conditional and direct edges between nodes
- Specify start and end conditions
- Compile the graph and validate its structure

#### Implementation Example
```python
# Set up the graph
graph_builder = StateGraph(KitchenState)

# Add all the nodes
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("human", human_node)
graph_builder.add_node("recipe_search", recipe_search_node)
graph_builder.add_node("recipe_detail", recipe_detail_node)
# Add the rest of the nodes...

# Define edges and conditional transitions
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "human")
graph_builder.add_conditional_edges("human", route_based_on_intent)
# Add the rest of the edges...

# Compile the graph
kitchen_assistant_graph = graph_builder.compile()
```

## 11. User Interface Integration

### Title: Integrate with User Interface

#### Description
Connect the LangGraph workflow with the user interface components.

#### Task Details
- Create handlers for text input from the UI
- Implement audio recording and processing for voice input
- Design output formatting for different response types
- Ensure state persistence across interactions

#### Implementation Notes
- Use the existing UI tabs for text and voice input
- Maintain conversation history in the UI
- Format recipe displays with appropriate styling
- Include visual indicators for processing state

## 12. Testing and Refinement

### Title: Test and Refine the Complete System

#### Description
Test the full system with various user scenarios and refine based on results.

#### Task Details
- Create test cases for different user intents
- Validate proper transitions between nodes
- Test error handling and recovery
- Tune prompts and system instructions based on observed behavior

#### Test Scenarios
1. Recipe search by ingredients
2. Recipe customization for dietary restrictions
3. Nutritional analysis requests
4. Voice input processing
5. Web-grounded answers to cooking questions

## Conclusion

This plan outlines the implementation of a comprehensive LangGraph workflow for our Kitchen Management Assistant. By following these steps, we'll create a powerful, stateful application that effectively leverages our existing components while adding the flexibility and robustness of a graph-based architecture.

The implementation will enable seamless user interactions across different modalities (text and voice) while providing valuable recipe discovery, customization, and nutrition analysis features.