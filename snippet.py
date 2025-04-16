@tool
def fetch_nutrition_from_openfoodfacts(ingredient_name: str) -> str:
    """
    Fetches nutrition data (per 100g) for a single ingredient from Open Food Facts API.
    Includes basic retry logic. Returns nutrition data as a JSON string or an error/unavailable status.
    """
    print(f"DEBUG TOOL CALL: fetch_nutrition_from_openfoodfacts(ingredient_name='{ingredient_name}')")
    search_url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {"search_terms": ingredient_name, "search_simple": 1, "action": "process", "json": 1, "page_size": 1}
    headers = {'User-Agent': 'KitchenAssistantLangGraph/1.0 (Language: Python)'}
    max_retries = 2
    retry_delay = 1

    for attempt in range(max_retries + 1): # Allow max_retries attempts
        try:
            response = requests.get(search_url, params=params, headers=headers, timeout=15)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if data.get('products') and len(data['products']) > 0:
                product = data['products'][0]
                nutriments = product.get('nutriments', {})
                # Extract only desired nutrient fields, handle missing values gracefully
                nutrition_info = {
                    "food_normalized": ingredient_name,
                    "source": "Open Food Facts",
                    "product_name": product.get('product_name', ingredient_name),
                    "calories_100g": nutriments.get('energy-kcal_100g'),
                    "fat_100g": nutriments.get('fat_100g'),
                    "saturated_fat_100g": nutriments.get('saturated-fat_100g'),
                    "carbohydrates_100g": nutriments.get('carbohydrates_100g'),
                    "sugars_100g": nutriments.get('sugars_100g'),
                    "fiber_100g": nutriments.get('fiber_100g'),
                    "proteins_100g": nutriments.get('proteins_100g'),
                    "sodium_100g": nutriments.get('sodium_100g'), # Sodium is often in mg, convert later if needed
                }
                # Filter out keys where value is None
                filtered_nutrition = {k: v for k, v in nutrition_info.items() if v is not None}

                # Check if at least one core nutrient is present
                core_nutrients = ["calories_100g", "fat_100g", "proteins_100g", "carbohydrates_100g"]
                if not any(k in filtered_nutrition for k in core_nutrients):
                     # Return unavailable if no core data found, even if product exists
                     return json.dumps({"status": "unavailable", "reason": f"No detailed nutrition data found for '{ingredient_name}'"})

                return json.dumps(filtered_nutrition, indent=2)
            else:
                # No product found
                return json.dumps({"status": "unavailable", "reason": f"No product found for '{ingredient_name}' on Open Food Facts"})

        except requests.exceptions.HTTPError as e:
            # Handle specific HTTP errors like rate limiting
            if e.response.status_code == 429 and attempt < max_retries:
                wait_time = (retry_delay * (2 ** attempt)) + random.uniform(0, 0.5)
                print(f"Rate limit hit for '{ingredient_name}'. Retrying in {wait_time:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue # Retry the loop
            else:
                # Other HTTP errors
                print(f"HTTP Error fetching nutrition for '{ingredient_name}': {e}")
                return json.dumps({"status": "unavailable", "reason": f"API request failed: {e}"})
        except requests.exceptions.RequestException as e:
            # Handle network-related errors (DNS, connection, timeout)
            if attempt < max_retries:
                 wait_time = (retry_delay * (2 ** attempt)) + random.uniform(0, 0.5)
                 print(f"Request error for '{ingredient_name}': {e}. Retrying in {wait_time:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                 time.sleep(wait_time)
                 continue # Retry the loop
            else:
                # Failed after retries
                print(f"Error fetching nutrition for '{ingredient_name}' after retries: {e}")
                return json.dumps({"status": "unavailable", "reason": f"API request failed after retries: {e}"})
        except json.JSONDecodeError:
            # Handle cases where the response is not valid JSON
            print(f"Error decoding JSON response for '{ingredient_name}'")
            return json.dumps({"status": "unavailable", "reason": "Invalid JSON response from API"})
        except Exception as e:
             # Catch any other unexpected errors
             print(f"ERROR in fetch_nutrition_from_openfoodfacts: {e}")
             import traceback
             traceback.print_exc()
             return json.dumps({"error": f"Unexpected error fetching nutrition for {ingredient_name}: {e}"})

    # If loop completes without success after all retries
    return json.dumps({"status": "unavailable", "reason": "Max retries exceeded for API request"})
