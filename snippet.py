import re
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

def extract_and_visualize_nutrition(response_text: str):
    """
    Extracts accumulated nutrition data from LLM response text and
    visualizes it as a color-coded horizontal bar chart (% Daily Value).

    Args:
        response_text: The string output from the LLM containing recipe details.
    """

    # --- 1. Extraction using Regex ---
    nutrition_section_match = re.search(
        r"\*\*Ingredient Nutrition:\*\*\s*\n(.*?)(?:\n\n|$)",
        response_text,
        re.DOTALL | re.IGNORECASE
    )

    if not nutrition_section_match:
        print("Could not find the 'Ingredient Nutrition:' section in the text.")
        return

    nutrition_text = nutrition_section_match.group(1).strip()

    # Regex to capture individual ingredient lines and their key-value pairs
    # Handles potential variations in spacing and key names slightly
    ingredient_pattern = re.compile(
        r"^\s*\*\s+\*\*(?P<ingredient>.*?):\*\*\s+(?P<data>.*?)$",
        re.MULTILINE
    )
    kv_pattern = re.compile(r"([\w_]+)\s*=\s*'?([^,']+)'?") # Capture key=value or key='value'

    accumulated_nutrition: Dict[str, float] = {
        "calories_100g": 0.0,
        "fat_100g": 0.0,
        "saturated_fat_100g": 0.0,
        "carbohydrates_100g": 0.0,
        "sugars_100g": 0.0,
        "fiber_100g": 0.0,
        "proteins_100g": 0.0,
        "sodium_100g": 0.0, # Keep in grams initially for consistency here
    }
    
    processed_ingredients = 0
    unavailable_ingredients = []
    # Track ingredient counts per nutrient for averaging
    nutrient_counts = {key: 0 for key in accumulated_nutrition.keys()}

    for match in ingredient_pattern.finditer(nutrition_text):
        ingredient_name = match.group("ingredient").strip()
        data_str = match.group("data").strip()

        # Check if this ingredient reported an error/unavailability
        if "status=" in data_str.lower() and "unavailable" in data_str.lower():
             unavailable_ingredients.append(ingredient_name)
             print(f"Skipping unavailable/error data for ingredient: {ingredient_name}")
             continue # Skip to next ingredient

        ingredient_data = dict(kv_pattern.findall(data_str))
        
        valid_data_found = False
        for key, value_str in ingredient_data.items():
            # Normalize key (remove _100g suffix if present for matching)
            norm_key = key.replace('_100g', '') + '_100g' 

            if norm_key in accumulated_nutrition:
                try:
                    # Convert value to float, handle potential errors
                    value = float(value_str)
                    # Only count non-zero values for water and other ingredients
                    if value > 0 or (ingredient_name.lower() != 'water' and norm_key != 'calories_100g'):
                        accumulated_nutrition[norm_key] += value
                        nutrient_counts[norm_key] += 1
                    valid_data_found = True
                except ValueError:
                    print(f"Warning: Could not convert value '{value_str}' for key '{key}' in ingredient '{ingredient_name}' to float. Skipping.")
            # else: # Uncomment if you want to see keys that weren't accumulated
            #     print(f"Info: Key '{key}' from '{ingredient_name}' not in accumulation list.")

        if valid_data_found:
             processed_ingredients += 1
             
    if processed_ingredients == 0:
         print("No valid nutrition data found to accumulate or plot.")
         if unavailable_ingredients:
              print(f"Unavailable ingredients: {', '.join(unavailable_ingredients)}")
         return

    # Calculate averages for each nutrient based on the number of ingredients that contributed values
    average_nutrition = {}
    for key, value in accumulated_nutrition.items():
        count = nutrient_counts[key]
        if count > 0:
            average_nutrition[key] = value / count
        else:
            average_nutrition[key] = 0.0
    
    print(f"Processed {processed_ingredients} ingredients.")
    if unavailable_ingredients:
         print(f"Note: Could not get data for: {', '.join(unavailable_ingredients)}")
    print("Average nutrition values:", average_nutrition)
    print("Nutrition counts (ingredients contributing to each value):", nutrient_counts)


    # --- 2. Normalization to % Daily Value (DV) ---
    # Reference Daily Values (adjust based on your target audience/standard, e.g., FDA)
    # Using FDA values as an example (approximated where needed)
    # Note: These DVs are for a *total daily diet*, using the average values
    #       per 100g of ingredient is more representative than using the sum.
    daily_values = {
        "calories_100g": 2000,  # kcal
        "fat_100g": 78,      # g
        "saturated_fat_100g": 20, # g
        "carbohydrates_100g": 275,# g
        "sugars_100g": 50,     # g (Reference for Added Sugars, using as proxy)
        "fiber_100g": 28,      # g
        "proteins_100g": 50,    # g
        "sodium_100g": 2.3,    # g (Note: DV is 2300mg = 2.3g)
    }

    percent_dv: Dict[str, float] = {}
    actual_values: Dict[str, float] = {} # Store the raw average values

    for key, avg_value in average_nutrition.items():
        dv = daily_values.get(key)
        if dv is not None and dv > 0:
            percent_dv[key] = round((avg_value / dv) * 100, 1)
            actual_values[key] = round(avg_value, 1)
        elif dv == 0 and avg_value == 0 : # Handle cases like 0 sodium DV if needed (though DV is non-zero)
             percent_dv[key] = 0.0 
             actual_values[key] = 0.0
        else:
            # Handle cases where DV isn't defined or is zero (shouldn't happen with above DVs)
            percent_dv[key] = 0.0 # Or handle as error/skip
            actual_values[key] = round(total_value, 1)
            print(f"Warning: No Daily Value defined or DV is zero for {key}. Cannot calculate %DV.")
            
    # Separate Calories as it has a different unit (kcal) and scale
    calories_percent_dv = percent_dv.pop("calories_100g", 0.0)
    calories_actual = actual_values.pop("calories_100g", 0.0)
    
    # Prepare data for plotting (nutrients other than calories)
    labels = list(percent_dv.keys())
    # Clean labels for display
    display_labels = [
        l.replace('_100g', '').replace('_', ' ').capitalize() 
        for l in labels
    ] 
    values = list(percent_dv.values())

    # --- 3. Color Coding ---
    colors = []
    # Define thresholds for %DV (adjust as needed)
    # Green: <= 50% DV
    # Orange: > 50% and <= 100% DV
    # Red: > 100% DV
    for v in values:
        if v <= 50:
            colors.append('forestgreen')
        elif v <= 100:
            colors.append('orange')
        else:
            colors.append('red')

    # --- 4. Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6)) # Adjusted figure size

    # Create horizontal bars
    bars = ax.barh(display_labels, values, color=colors, height=0.6)

    # Add labels and title
    ax.set_xlabel('% Daily Value (DV) - Based on sum of 100g of each ingredient')
    ax.set_title('Accumulated Ingredient Nutrition (%DV)', fontsize=16)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)

    # Add value labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        # Get the actual gram/mg value for annotation
        nutrient_key = labels[i]
        actual_val = actual_values.get(nutrient_key, 0.0)
        unit = 'mg' if nutrient_key == 'sodium_100g' else 'g'
        if nutrient_key == 'sodium_100g':
            actual_val *= 1000 # Convert sodium back to mg for display clarity
            
        label_text = f'{width:.1f}% ({actual_val:.1f} {unit})'
        
        # Position label - inside if bar is long enough, otherwise outside
        x_pos = width + 1 if width < 90 else width - 1 # Adjust positioning threshold
        ha = 'left' if width < 90 else 'right'
        color = 'black' if width < 90 else 'white'

        ax.text(x_pos, bar.get_y() + bar.get_height()/2., label_text,
                ha=ha, va='center', color=color, fontsize=9, fontweight='bold')

    # Add Calorie Information separately
    cal_color = 'forestgreen' if calories_percent_dv <= 50 else ('orange' if calories_percent_dv <= 100 else 'red')
    calorie_text = f'Estimated Calories Sum: {calories_actual:.0f} kcal ({calories_percent_dv:.1f}% DV)'
    # Add text annotation for calories at the top or bottom
    fig.text(0.5, 0.95, calorie_text, ha='center', va='top', fontsize=12, color=cal_color, fontweight='bold')

    # Adjust layout and display
    plt.gca().invert_yaxis() # Display top-to-bottom typically looks better
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Instead of plt.show(), you might want to save the figure in a web app context
    # plt.savefig('nutrition_chart.png')
    plt.show()




# Run the function with the example text
extract_and_visualize_nutrition(response.text)