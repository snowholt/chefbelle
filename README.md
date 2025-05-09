<div align="center">
    <img src="https://i.ibb.co/svNkqDC9/logo.png" alt="Chefbelle Logo" width="350">
</div>

# ✨ Chefbelle: Your Interactive AI Kitchen Assistant ✨

Chefbelle is an innovative AI-powered kitchen assistant designed to tackle the everyday dilemma: **"What can I cook with the ingredients I actually have?"**. It moves beyond traditional recipe apps by understanding your available ingredients, dietary needs, and preferences to suggest personalized recipes and provide interactive cooking guidance.

**Developers:** Nariman Jafarieshlaghi, Neda Saberitabar.

---

## 👨‍💻 Contributors

- **Nariman Jafarieshlaghi** - Lead AI Engineer
- **Neda Saberitabar** - Data Scientist & ML Engineer
- **You?** - We welcome contributions! See our [contribution guidelines](#contributing) below.

---

## 🍳 The Problem: The Everyday Kitchen Dilemma

We've all been there: staring into the fridge at a random assortment of ingredients – half an onion, some leftover chicken, a lonely bell pepper – wondering what meal we can possibly create. Traditional recipe apps often assume a fully stocked pantry, offer rigid instructions, and don't adapt to specific needs or available time. This often leads to food waste, mealtime stress, and reliance on takeout.

## 🥗 Introducing Chefbelle: Cooking, Reimagined

Chefbelle aims to transform your cooking experience by being a smart, interactive kitchen companion. It helps you:

1.  **Discover:** Find delicious recipes based on the ingredients you *already have*.
2.  **Customize:** Adapt meals to fit dietary goals (Vegan, Gluten-Free, Low-Carb, etc.) and preferences.
3.  **Guide:** Receive clear, step-by-step cooking instructions.
4.  **Inform:** Understand the nutritional impact of your meals using reliable data.
5.  **Interact:** Use natural language via voice or text commands.

**Our Goal:** To build an intuitive and genuinely helpful kitchen assistant that empowers home cooks, reduces food waste, and makes cooking more enjoyable and personalized.

---

## 🧠 Technical Stack & Architecture

Chefbelle leverages a suite of cutting-edge Generative AI capabilities and related technologies:

*   **LLM Core:** **Google Gemini** (specifically `gemini-1.5-flash` in this implementation) powers natural language understanding, intent recognition, response generation, and function calling orchestration.
*   **Agent Framework:** **LangGraph** is used to build the stateful, multi-step agent, managing the conversation flow, tool usage, and state transitions.
*   **Tools / Capabilities:**
    *   **Database Querying (SQLite):** Tools to `list_tables`, `describe_table`, and `execute_query` (read-only) on a structured database (`kitchen_db.sqlite`) containing recipe details, interactions, and nutrition data. Includes specific tools like `get_recipe_by_id` and `get_ratings_and_reviews_by_recipe_id`.
    *   **Semantic Search (ChromaDB):** A vector database stores recipe embeddings, enabling `gemini_recipe_similarity_search` to find recipes based on conceptual similarity, not just keywords. Includes filtering capabilities. An `gemini_interaction_similarity_search` tool is also available for review analysis.
    *   **External Nutrition API (USDA FDC):** The `fetch_nutrition_from_usda_fdc` tool calls the official USDA FoodData Central API for reliable, standardized nutritional information per ingredient (with built-in retries).
    *   **Grounding (Google Search):** Gemini's built-in grounding feature leverages Google Search for real-time answers to general cooking questions outside the scope of the structured data.
    *   **Audio Understanding (Speech-to-Text):** The `transcribe_audio` function supports **Google Cloud Speech-to-Text** or **OpenAI Whisper** (tested with Google in the notebook) to process voice commands.
    *   **Structured Output (JSON):** Tools often return results in JSON format for reliable parsing by the agent/nodes.
    *   **(Placeholders):** `fetch_live_recipe_data` (simulates web scraping) and `customize_recipe` (simulates recipe modification logic) are included as placeholders for future development.
*   **Data Handling:** **Pandas** for loading, preprocessing, and manipulating datasets.
*   **Visualization:** **Matplotlib** and **Seaborn** for data exploration and nutrition visualization.
*   **UI Simulation:** **ipywidgets** to create a basic interactive interface within the Jupyter notebook for testing.
*   **Stateful Conversation:** The LangGraph `KitchenState` schema maintains context across turns, including message history, selected recipe, processed tool outputs, etc.
*   **Sentiment Analysis (Optional):** **VADER** library integration for analyzing review sentiment within the `ReviewDashboardNode`.

---

## 📊 Data

Chefbelle's knowledge is built upon the following datasets (primarily sourced from Kaggle):

1.  **Food.com Recipes:** (~230k recipes) Contains recipe ID, name, description, ingredients, steps, tags, cooking time, contributor info, etc.
2.  **Food.com Interactions:** (~1.1M interactions) Includes user ID, recipe ID, rating, and text reviews.
3.  **Nutritional Breakdown of Foods:** (~3.5k records) Provides nutritional details for common food items (used for fallback/exploration).
4.  **Food.com Vectorized (ChromaDB):** A pre-computed ChromaDB vector database containing embeddings for the recipes, enabling fast similarity search. Loaded directly from Kaggle datasets.

**Preprocessing Steps:** The notebook includes steps for:
*   Parsing list-like columns (ingredients, steps, tags).
*   Removing duplicate recipes.
*   Normalizing ingredient names (lowercase, removing quantities).
*   Basic dietary tag identification (vegetarian, gluten-free, etc.) based on keywords.

---

## ⚙️ Setup & Installation

To run this project locally (outside Kaggle):

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/chefbelle.git
    cd chefbelle
    ```
2.  **Create Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or install manually based on notebook imports:
    # pip install google-generativeai langchain-google-genai langgraph chromadb pandas matplotlib seaborn requests ipywidgets beautifulsoup4 openai google-cloud-speech vaderSentiment pydub soundfile
    ```
    *(Note: Ensure compatible versions as specified in the notebook, especially for `google-generativeai`, `langgraph`, and `chromadb`)*

4.  **API Keys:**
    *   Obtain API keys for:
        *   Google Gemini (from [AI Studio](https://aistudio.google.com/app/apikey))
        *   USDA FoodData Central (from [FDC API site](https://fdc.nal.usda.gov/api-key-signup.html))
        *   (Optional) OpenAI API Key (if using Whisper)
        *   (Optional) Google Cloud Credentials JSON (if using Google Cloud Speech-to-Text - requires project setup, API enabling, service account creation).
    *   **Securely store these keys.** Using environment variables is recommended:
        ```bash
        export GOOGLE_API_KEY="your_gemini_key"
        export USDA_API_KEY="your_usda_key"
        export OPENAI_API_KEY="your_openai_key"
        export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google_credentials.json"
        ```
        Alternatively, create a `.env` file and use a library like `python-dotenv`. **Do not commit keys directly into code.** The notebook uses Kaggle Secrets, which needs adaptation for local use.

5.  **Data & Databases:**
    *   Download the required datasets from Kaggle:
        *   [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
        *   [Nutritional Breakdown of Foods](https://www.kaggle.com/datasets/datasciencebowl/cleaned_nutrition_dataset) (or similar)
        *   [Food.com Vectorized with ChromaDB](https://www.kaggle.com/datasets/narimanjafarieshlaghi/food-com-vectorized-with-chromadb) (Pre-built Vector DB)
    *   Place the CSV files and the extracted `vector_db` directory in appropriate locations accessible by the notebook (e.g., an `input/` directory).
    *   Update the `VECTOR_DB_PATH` and CSV loading paths in the notebook if necessary.
    *   The notebook will create the `kitchen_db.sqlite` file in a `final/` directory upon running the database setup cells.

---

## ▶️ Usage

1.  Ensure all setup steps are complete (environment, dependencies, API keys, data).
2.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook capstone-2025-kma-nn.ipynb
    # or
    jupyter lab capstone-2025-kma-nn.ipynb
    ```
3.  Run the cells sequentially. Pay attention to:
    *   **API Key Setup:** Ensure the code correctly loads your keys (modify the Kaggle Secrets part).
    *   **Data Loading:** Verify paths to CSVs and the ChromaDB directory.
    *   **Database Setup:** The SQLite DB will be created. The ChromaDB setup is commented out as it uses pre-existing data.
4.  Interact with the agent using the **Simulated Interface** cells (Phase 9) or the **Chat Helper Functions** (Phase 10).

---

## 🚀 Future Work

*   Implement the placeholder tools (`fetch_live_recipe_data`, `customize_recipe`) with actual logic (web scraping, advanced LLM calls).
*   Persist user context (ingredients, preferences) across sessions.
*   Develop a more robust web application interface (e.g., using Streamlit, Flask, or Gradio).
*   Improve ingredient normalization and dietary tag accuracy.
*   Enhance error handling and user feedback.
*   Implement more sophisticated recipe recommendation logic.
*   Deploy the assistant.

---

## 📄 License

[Specify your license here, e.g., MIT License, Apache 2.0] (Defaults to None if not specified)
