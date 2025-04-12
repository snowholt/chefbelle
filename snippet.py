import google.generativeai as genai
from google.generativeai import types
import os
import re
import pandas as pd
from IPython.display import display, Markdown, Code # Add Code import

# Assume GOOGLE_API_KEY is loaded from environment or secrets
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

genai.configure(api_key=GOOGLE_API_KEY)

# --- Part 1: Generate Nutritional Summary Text ---

# Instantiate the model for text generation
text_model = genai.GenerativeModel("gemini-2.0-flash")

nutritional_data_prompt = """Calculate the total nutritional values of a recipe based on the available data for its ingredients. Follow these rules:

1. For each ingredient listed, use the given nutritional information (per 100g).
2. If the API request failed or the nutritional info is unavailable for an ingredient, ignore it completely.
3. Sum each nutritional column (e.g., calories, carbohydrates, fat, etc.) **across only the ingredients with available data**.
4. After summing, divide each value by the number of ingredients that had valid data to get an average per 100g.
5. Present the result ONLY as a single sentence in the format: "RECIPE_NAME contains approximately: X calories, Y carbohydrates, Z fat, ... per 100g." Do not include any other text, titles, or explanations.
6. Use the recipe name: “Country White Bread or Dinner Rolls”

Here is the nutritional data (per 100g, from Open Food Facts):

water: API request failed
egg: calories_100g: 725, carbohydrates_100g: 1.4, fat_100g: 79, proteins_100g: 1.1, saturated_fat_100g: 6.3, sodium_100g: 0.6, sugars_100g: 1.3
vegetable: calories_100g: 675, carbohydrates_100g: 0.2, fat_100g: 75, fiber_100g: 0, proteins_100g: 0.1, saturated_fat_100g: 34, sodium_100g: 0.2, sugars_100g: 0
oil: API request failed
bread: API request failed
flour: calories_100g: 116, carbohydrates_100g: 16.5, fat_100g: 5.1, fiber_100g: 0.1, proteins_100g: 1, saturated_fat_100g: 2.5, sodium_100g: 0.02, sugars_100g: 9.2
sugar: calories_100g: 116, carbohydrates_100g: 16.5, fat_100g: 5.1, fiber_100g: 0.1, proteins_100g: 1, saturated_fat_100g: 2.5, sodium_100g: 0.02, sugars_100g: 9.2
salt: carbohydrates_100g: 0, fat_100g: 0, fiber_100g: 0, proteins_100g: 0, saturated_fat_100g: 0, sodium_100g: 39.6, sugars_100g: 0
instant: calories_100g: 515, carbohydrates_100g: 63, fat_100g: 26, fiber_100g: 1, proteins_100g: 7.1, saturated_fat_100g: 7.74, sodium_100g: 0.098, sugars_100g: 61
yeast: calories_100g: 260, carbohydrates_100g: 30, fat_100g: 0.5, fiber_100g: 0, proteins_100g: 34, saturated_fat_100g: 0.1, sodium_100g: 4.32, sugars_100g: 1.2
butter: calories_100g: 675, carbohydrates_100g: 0.2, fat_100g: 75, fiber_100g: 0, proteins_100g: 0.1, saturated_fat_100g: 34, sodium_100g: 0.2, sugars_100g: 0
shortening: calories_100g: 368, carbohydrates_100g: 49.12, fat_100g: 17.54, fiber_100g: 1.8, proteins_100g: 5.26, saturated_fat_100g: 5.26, sodium_100g: 0.333, sugars_100g: 17.54
"""

# Use the model directly
response_text_gen = text_model.generate_content(
    # model="gemini-2.0-flash", # Model specified during instantiation
    contents=nutritional_data_prompt
)

nutritional_summary_text = response_text_gen.text
print("Generated Nutritional Summary:")
print(nutritional_summary_text)
print("-" * 30)


# --- Part 2: Visualize the Nutritional Summary using Code Generation --- # Renamed section

# Instantiate the model WITHOUT the code execution tool
chat_model = genai.GenerativeModel(
    "gemini-2.0-flash" # Removed tools=['code_execution']
)

# Create a chat session using the model's start_chat method
chat = chat_model.start_chat(history=[])

# Construct the prompt to ask for code generation
visualization_prompt = f"""
You are given the following nutritional summary text:
"{nutritional_summary_text}"

Your task is to generate Python code that performs the following steps:
1. Parses the text to extract the nutritional values (calories, carbohydrates, fat, fiber, proteins, saturated fat, sodium, sugars) and their corresponding amounts per 100g. Handle potential variations in nutrient names (e.g., 'proteins' vs 'protein').
2. Creates a pandas DataFrame with two columns: 'Nutrient' and 'Amount (per 100g)'.
3. Generates a bar chart using seaborn or matplotlib to visualize these nutritional amounts.
4. Labels the x-axis 'Nutrient' and the y-axis 'Amount (per 100g)'.
5. Adds a title to the plot, for example: "Nutritional Breakdown for [Recipe Name]". Extract the recipe name from the input text.
6. Ensures the plot is displayed using `plt.show()`.
7. Includes all necessary imports (pandas, seaborn, matplotlib.pyplot, re) within the generated code block.

Output ONLY the complete Python code block required to perform these steps, enclosed in triple backticks (```python ... ```). Do not include any explanatory text before or after the code block.
"""

# Send the prompt to the chat session to generate the code
response_viz = chat.send_message(
    content=visualization_prompt,
)

# Extract and display the generated Python code block
generated_code_text = response_viz.text
# Simple extraction assuming the model follows the ```python ... ``` format
match = re.search(r"```python\n(.*)\n```", generated_code_text, re.DOTALL)
if match:
    generated_code = match.group(1).strip()
    print("--- Generated Visualization Code ---")
    # Display as a formatted code block if in an environment like Jupyter/Colab
    try:
        display(Code(generated_code, language='python'))
    except NameError: # Fallback for non-IPython environments
        print(generated_code)
    print("-" * 30)
else:
    print("--- Model Response (Could not extract code block) ---")
    # Display the raw response if extraction failed
    display(Markdown(generated_code_text))
    print("-" * 30)