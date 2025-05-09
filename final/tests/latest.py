#!/usr/bin/env python
# coding: utf-8

# <div align="center">
#     <img src="https://i.ibb.co/svNkqDC9/logo.png" alt="Chefbelle Logo" width="350">
# </div>
# 
# 
# # ✨ Chefbelle: Your Interactive AI Kitchen Assistant ✨
# 
# ## The Everyday Kitchen Dilemma
# 
# Picture this: You open the fridge, stare at a random assortment of ingredients – half an onion, some leftover chicken, a lonely bell pepper – and the familiar question echoes: **"What on earth can I cook with *this*?"**
# 
# Traditional recipe apps often fall short. They assume a fully stocked pantry, offer static instructions, and rarely cater to *your* specific dietary needs, available time, or the ingredients you *actually* have. This leads to food waste, mealtime stress, and maybe one too many takeout orders. 😅
# 
# ## Introducing Chefbelle: Cooking, Reimagined
# 
# What if you had a smart kitchen companion who *understood* your reality? Meet **Chefbelle**, an AI-powered assistant designed to transform your cooking experience. Chefbelle helps you:
# 
# 1.  🍳 **Discover** delicious recipes based on the ingredients you *already have*.
# 2.  🥗 **Customize** meals to fit your dietary goals (Vegan? Gluten-free? Low-carb? No problem!).
# 3.  📝 **Receive** clear, step-by-step cooking guidance.
# 4.  📊 **Understand** the nutritional impact of your meals.
# 5.  🗣️ **Interact** naturally using voice or text commands.
# 
# ## Powered by Generative AI
# 
# Chefbelle leverages a suite of cutting-edge Generative AI capabilities to provide a seamless and intelligent experience:
# 
# *   🧠 **LLM Core (Gemini):** Powers natural language understanding and response generation.
# *   🗣️ **Audio Understanding:** Processes voice commands via Speech-to-Text (Whisper/Google Speech).
# *   🔧 **Function Calling:** Enables Chefbelle to interact with databases and APIs (like USDA Nutrition).
# *   🤖 **Agents (LangGraph):** Orchestrates complex conversational flows and tool usage.
# *   🌐 **Grounding:** Uses Google Search for real-time answers to general cooking questions.
# *   🔍 **Embeddings & Vector DB (ChromaDB):** Enables semantic search for finding relevant recipes.
# *   📚 **Retrieval Augmented Generation (RAG):** Fetches recipe and nutrition data to ground responses in facts, reducing hallucination.
# *   ⚙️ **Structured Output (JSON):** Facilitates reliable data exchange between the agent and its tools.
# *   💾 **Stateful Conversation:** Remembers context across turns for a natural dialogue.
# *   📈 **GenAI Evaluation:** Focuses on reasoned, data-driven responses over simple generation.
# 
# **Our Goal:** To build Chefbelle – an innovative, intuitive, and genuinely helpful kitchen assistant that empowers home cooks, reduces food waste, and makes cooking more enjoyable and personalized. This notebook documents the journey of bringing Chefbelle to life, step by step.
# 
# 
# ---
# 
# ## 👩‍💻 Developers
# 
# <div align="center">
#   <a href="https://github.com/snowholt" target="_blank">
#     <img src="https://img.shields.io/badge/Nariman%20Jafarieshlaghi-%23181717?style=for-the-badge&logo=github&logoColor=white" alt="Nariman GitHub"/>
#   </a>
#   <br/><br/>
#   <a href="https://github.com/Nedasaberitabar" target="_blank">
#     <img src="https://img.shields.io/badge/Neda%20Saberitabar-%23181717?style=for-the-badge&logo=github&logoColor=white" alt="Neda GitHub"/>
#   </a>
# </div>
# 
# ---
# 
# ## 🔗 Project Links
# 
# - 🌐 **Website:** [www.chefbelle.com](https://www.chefbelle.com)
# - 💻 **GitHub Repo:** [github.com/snowholt/chefbelle](https://github.com/snowholt/chefbelle)
# - 📺 **YouTube Demo:** [Watch here](https://youtu.be/9VqDCcgHW-g)
# 

# 
# > ## "Good code is like a good joke – it needs no explanation."
# > — Gregor Hohpe
# 
# *   However... since we're still mastering the art of AI and cooking up Chefbelle, we'll add helpful comments along the way.
# *   Think of them as subtitles for the journey – making sure every step is clear, even as we build something complex! 😉
# 

# ## Phase 1: Gathering Our Ingredients - Data Foundation
# 
# Every great recipe starts with quality ingredients. For Chefbelle, our "ingredients" are data. In this first phase, we'll acquire, explore, and prepare the foundational recipe dataset. This data will fuel Chefbelle's knowledge base, enabling recipe retrieval, understanding, and customization. Let's get cooking with data! 📊

# ## 1.1. Setting the Stage: Preparing Our Workspace
# 
# Before we can start building Chefbelle, we need to set up our digital kitchen – the development environment. This involves installing the necessary Python libraries, ensuring compatibility, and importing the tools we'll use throughout the project.
# 
# *This cell handles the installation and potential cleanup of required libraries like `google-generativeai`, `langgraph`, `chromadb`, `pandas`, and more. Ensuring the right versions are installed prevents unexpected issues later.*

# In[1]:


# Clean up and install compatible versions
#!pip uninstall -y tensorflow protobuf google-api-core google-cloud-automl google-generativeai google-cloud-translate chromadb
get_ipython().system('pip uninstall -qqy kfp > /dev/null 2>&1')

# Install chromadb with compatible versions
get_ipython().system('pip install -qU --no-warn-conflicts "google-genai==1.7.0" chromadb==0.6.3')
# #!pip install -U google-api-core==2.16.0

get_ipython().system('pip install -q --no-warn-conflicts google-cloud-speech')

# Install base packages with minimal dependencies
get_ipython().system('pip install -q --no-warn-conflicts pandas matplotlib seaborn')
get_ipython().system('pip install -q --no-warn-conflicts kagglehub[pandas-datasets]')
get_ipython().system('pip install -q --no-warn-conflicts soundfile pydub ipywidgets openai')

# Install compatible versions
#!pip install -q google-generativeai  # Latest version instead of 1.7.0




# ### Install LangGraph and Dependencies
# 
# Uninstalls potentially conflicting packages from the base environment and installs specific versions of `langgraph`, `langchain-google-genai`, and `langgraph-prebuilt` required for the agent implementation.

# In[2]:


# Remove conflicting packages from the Kaggle base environment.
get_ipython().system('pip uninstall -qqy kfp libpysal thinc spacy fastai ydata-profiling google-cloud-bigquery google-generativeai')
# Install langgraph and the packages used in this lab.
get_ipython().system("pip install -qU 'langgraph==0.3.21' 'langchain-google-genai==2.1.2' 'langgraph-prebuilt==0.1.7'")


# ### LangGraph Step 1: Define State Schema (`KitchenState`)
# 
# Defines the `TypedDict` class `KitchenState` which represents the shared memory or state of the LangGraph agent. It includes fields for message history, user input, parsed intent, context (like selected recipe ID), raw tool outputs, processed data, user preferences, and control flow flags. An initial state dictionary is also defined.
# 
# 

# In[3]:


get_ipython().system('pip install graphviz')


# ## 1.2. Importing Our Tools: Libraries & Utilities
# 
# With the environment ready, let's import all the necessary Python libraries. We've organized them by function – from data handling (Pandas) and visualization (Matplotlib) to AI models (Gemini, LangChain/LangGraph) and database interactions (ChromaDB, SQLite). This keeps our workspace tidy and makes the code easier to follow.

# In[4]:


# 
get_ipython().run_line_magic('matplotlib', 'inline')


# ===============================
# 📦 Standard Library Imports
# ===============================
import contextlib
import datetime
import io
import os
import random
import re
import sqlite3
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Tuple
from io import StringIO

# ===============================
# 📊 Data Analysis & Visualization
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 🌐 Networking & APIs
# ===============================
import requests
import json

# ===============================
# 🧠 Widgets & Interactive Display
# ===============================
import ipywidgets as widgets
from IPython.display import Audio, Image, Markdown, clear_output, display, HTML

# ===============================
# 🔍 ChromaDB Vector DB
# ===============================
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ===============================
# 💬 LangChain / LangGraph
# ===============================
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# ===============================
# 🤖 Google Gemini & Speech APIs
# ===============================
from google import genai
from google.api_core import retry
from google.genai import types

# Retry mechanism for Gemini API
is_retriable = lambda e: (
    isinstance(e, genai.errors.APIError) and e.code in {429, 503}
)
genai.models.Models.generate_content = retry.Retry(predicate=is_retriable)(
    genai.models.Models.generate_content
)

# ===============================
# 🎤 Audio Processing (Optional)
# ===============================
try:
    import soundfile as sf
    import sounddevice as sd
    AUDIO_LIBRARIES_AVAILABLE = True
    print("Audio libraries imported successfully!")
except (ImportError, OSError) as e:
    AUDIO_LIBRARIES_AVAILABLE = False
    print(f"Warning: Audio libraries could not be imported: {e}")

# ===============================
# 🗣️ Google Cloud Speech‑to‑Text (Optional)
# ===============================
try:
    from google.cloud import speech
    GOOGLE_SPEECH_AVAILABLE = True
    print("Google Cloud Speech‑to‑Text is imported successfully!")
except ImportError:
    GOOGLE_SPEECH_AVAILABLE = False
    print(
        "Google Cloud Speech‑to‑Text not available. "
        "Will use simulation for speech recognition."
    )

# ===============================
# 🎨 Visualization & Display Setup
# ===============================
plt.style.use("ggplot")
sns.set(style="whitegrid")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 50)
pd.set_option("display.max_colwidth", 100)

print("Environment setup complete!")
print("Google Gemini version:", genai.__version__)


# --------------------------------
# Extra helper / utility imports
# --------------------------------
from typing import Dict, Optional  # additional hints for helpers
from langgraph.graph import END   # used by visualize_nutrition_node
from langgraph.graph import StateGraph, START, END

# ---> ADDED Markdown display <---
from IPython.display import display, clear_output, Markdown

# Sentiment‑analysis placeholder (optional)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
except ImportError:
    print(
        "Warning: vaderSentiment library not found. "
        "pip install vaderSentiment for review sentiment analysis."
    )
    analyzer = None


# ## 1.3. Defining Storage Locations: Database Paths
# 
# Chefbelle needs places to store her knowledge. We'll define the file paths for:
# 1.  **ChromaDB:** A vector database to store recipe embeddings for fast semantic search ("find recipes like...").
# 2.  **SQLite:** A traditional database for structured recipe details, user interactions, and nutritional information.

# In[5]:


# Define paths for ChromaDB and SQL database
VECTOR_DB_PATH = "/kaggle/input/food-com-vectorized-with-chromadb/vector_db"
DB_PATH = "final/kitchen_db.sqlite"


# ## 1.4 Access Credentials: Setting Up API Keys
# 
# To unlock the power of Google's Generative AI (Gemini), Google Cloud Speech, USDA Nutrition API, and potentially others (like OpenAI for Whisper), Chefbelle needs secure API keys.
# 
# If you don't already have an API key, you can grab one from [AI Studio](https://aistudio.google.com/app/apikey). You can find [detailed instructions in the docs](https://ai.google.dev/gemini-api/docs/api-key).
# 
# 🔑 **Security First:** We're using Kaggle Secrets to securely store these keys. *Never* hardcode API keys directly in your notebook!
# 
# *   `GOOGLE_API_KEY`: For Gemini API access.
# *   `GOOGLE_APPLICATION_CREDENTIALS`: A path to (or JSON content of) a service account key file, required for Google Cloud services like Speech-to-Text. API keys alone are often insufficient for Cloud services.
# *   `USDA_API_KEY`: For accessing the FoodData Central nutrition database.
# *   `OPENAI_API_KEY`: (Optional) If using OpenAI's Whisper for transcription.
# 
# *This cell retrieves the keys from Kaggle Secrets.*

# In[6]:


from kaggle_secrets import UserSecretsClient

GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
OPENAI_API_KEY = UserSecretsClient().get_secret("OPENAI_API_KEY")
SecretValueJson = UserSecretsClient().get_secret("GOOGLE_APPLICATION_CREDENTIALS") # Use the label you gave the secret
USDA_API_KEY = UserSecretsClient().get_secret("USDA_API_KEY")


# ## Phase 2: Loading the Pantry - Acquiring Recipe Data
# 
# Now, let's stock Chefbelle's pantry with data! We'll load the datasets that form the core of her knowledge:
# 
# 1.  **Food.com Recipes:** Contains detailed information about hundreds of thousands of recipes (ingredients, steps, descriptions, tags).
# 2.  **Food.com Interactions:** Includes user ratings and reviews, crucial for understanding recipe popularity and feedback.
# 3.  **Nutritional Data:** A separate dataset providing nutritional breakdowns for common food items (used for few-shot prompting examples or fallback).
# 
# We'll load these directly from Kaggle datasets. The code also includes essential post-processing steps, like converting string representations of lists (e.g., ingredients, steps) into actual Python lists, calculating the number of ingredients, and deriving a basic `cuisine_type`.

# In[7]:


# Option 1: Direct Kaggle dataset import
# This is the easiest way to import datasets in Kaggle notebooks

try:
    # If the dataset is added via the "Add data" button, it will be available at /kaggle/input/
    recipes_df = pd.read_csv('/kaggle/input/food-com-recipes-and-user-interactions/RAW_recipes.csv')
    interactions_df = pd.read_csv('/kaggle/input/food-com-recipes-and-user-interactions/RAW_interactions.csv')
    nutrition_df = pd.read_csv('/kaggle/input/nutritional-breakdown-of-foods/cleaned_nutrition_dataset.csv')
    print(f"Successfully loaded {len(recipes_df)} recipes")
    print(f"Successfully loaded {len(interactions_df)} interactions")
    print(f"Successfully loaded nutritional dataset with {len(nutrition_df)} records")




except FileNotFoundError:
    print("Dataset files not found. Please make sure you've added the dataset to your Kaggle notebook.")
    print("You can add it by clicking the 'Add data' button in the right sidebar.")
    print("Alternatively, you can use direct URLs if available.")

# Let's parse the JSON strings in the columns that contain lists
if 'recipes_df' in locals():
    # Check the actual structure of the dataframe

    # For Food.com dataset, ingredients, steps, and tags are stored as strings that represent lists
    # We need to convert them from string representation to actual Python lists
    try:
        if 'ingredients' in recipes_df.columns:
            recipes_df['ingredients'] = recipes_df['ingredients'].apply(eval)
            print("Successfully parsed ingredients column")

        if 'steps' in recipes_df.columns:
            recipes_df['steps'] = recipes_df['steps'].apply(eval)
            print("Successfully parsed steps column")

        if 'tags' in recipes_df.columns:
            recipes_df['tags'] = recipes_df['tags'].apply(eval)
            print("Successfully parsed tags column")

            # Add cuisine type based on tags
            recipes_df['cuisine_type'] = recipes_df['tags'].apply(
                lambda x: next((tag for tag in x if tag in ['italian', 'persian', 'mexican', 'chinese', 'indian', 'french', 'thai']), 'other')
            )


        # Count number of ingredients
        recipes_df['n_ingredients'] = recipes_df['ingredients'].apply(len)

        print("\nDataset successfully processed")

    except Exception as e:
        print(f"Error processing dataset: {e}")
        print("Column sample values:")
        for col in recipes_df.columns:
            print(f"{col}: {recipes_df[col].iloc[0]}")

print(f"\n{'-' * 30}")


# ## Phase 3: Tasting the Ingredients - Data Exploration & Understanding
# 
# Before Chefbelle can use the data, we need to understand it – just like tasting ingredients before cooking! This phase involves exploring the structure, content, and quality of our datasets.
# 
# We'll check:
# *   Data types (Are numbers stored as numbers? Text as text?)
# *   Missing values (Are there gaps in the data we need to handle?)
# *   Basic statistics (What's the typical cooking time? How many ingredients per recipe?)
# *   Distributions (What are the most common cuisines? How many reviews per recipe?)
# 
# This exploration helps us identify necessary cleaning steps and informs how Chefbelle can best utilize the information.

# ## 3.1. Inspecting the Goods: DataFrame Analysis Function
# 
# This function provides a systematic way to analyze our DataFrames, printing key information and generating plots to visualize distributions and missing data.

# In[8]:


# Function to analyze dataframe properties with visualizations
def analyze_dataframe(df, df_name):
    print(f"\n{'-' * 30}")
    print(f"Analysis for {df_name}:")
    print(f"{'-' * 30}")

    # Check data types
    print("\nData types:")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")

    # Missing values
    print("\nMissing values per column:")
    missing_values = df.isnull().sum()
    has_missing = missing_values[missing_values > 0]
    if not has_missing.empty:
        for col, missing in has_missing.items():
            print(f"{col}: {missing} missing values ({missing/len(df):.2%})")

        # 🔴 Plot missing values
        plt.figure(figsize=(8, 4))
        sns.barplot(x=has_missing.index, y=has_missing.values)
        plt.title(f"Missing Values in {df_name}")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No missing values")

    # Summary statistics
    print("\nNumeric columns summary:")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_cols:
        stats = df[numeric_cols].describe().T[['count', 'mean', 'min', 'max']]
        print(stats)

        # 📊 Histograms of numeric columns
        df[numeric_cols].hist(bins=30, figsize=(15, 10), color='lightblue', edgecolor='black')
        plt.suptitle(f"Distributions of Numeric Columns - {df_name}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        print("No numeric columns found.")

print("\n=== DATA ANALYSIS FOR ALL DATAFRAMES ===")
analyze_dataframe(recipes_df, "Recipes")
analyze_dataframe(interactions_df, "Interactions")
analyze_dataframe(nutrition_df, "Nutrition")


# In[ ]:





# ## 3.2. Display Sample Recipe Data
# 
# Prints a random sample of 3 rows from the `recipes_df` DataFrame to provide a quick look at the data structure and content.

# In[9]:


# Sample a few rows instead of full stats
print("\nSample rows:")
print(recipes_df.sample(3))


# In[ ]:





# ## 3.3. Flavor Profiles: Visualizing Cuisine Distribution 🍜
# 
# What types of cuisines are most common in our dataset? A quick visualization helps us understand the variety Chefbelle will be working with.

# In[10]:


# Top 15 cuisine types
if 'cuisine_type' in recipes_df.columns:
    top_cuisines = recipes_df['cuisine_type'].value_counts().nlargest(15)
    df_top = top_cuisines.reset_index()
    df_top.columns = ['cuisine', 'count']

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_top,
        x='cuisine',
        y='count',
        hue='cuisine',
        dodge=False,
        palette=sns.color_palette("husl", len(df_top)),
    )

    plt.legend().remove()  # 👈 remove the legend here

    plt.title('Top 15 Cuisine Types', fontsize=16, fontweight='bold')
    plt.xlabel('Cuisine Type', fontsize=12)
    plt.ylabel('Number of Recipes', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# In[11]:


# # Distribution of cooking time - use smaller bins
# if 'cooking_time' in recipes_df.columns:
#     plt.figure(figsize=(10, 6))
#     # Use log scale for better visualization if the range is large
#     if recipes_df['cooking_time'].max() > 5 * recipes_df['cooking_time'].median():
#         sns.histplot(recipes_df['cooking_time'].clip(upper=recipes_df['cooking_time'].quantile(0.95)), bins=20)
#         plt.title('Distribution of Cooking Time (minutes) - Clipped at 95th percentile')
#     else:
#         sns.histplot(recipes_df['cooking_time'], bins=20)
#         plt.title('Distribution of Cooking Time (minutes)')
#     plt.xlabel('Cooking Time (minutes)')
#     plt.ylabel('Count')
#     plt.tight_layout()
#     plt.show()


# ## 3.4. Recipe Complexity: Visualizing Ingredient Counts 🥕
# 
# How many ingredients do most recipes call for? This gives us insight into the typical complexity users might encounter. We'll visualize the distribution.

# In[12]:


if 'n_ingredients' in recipes_df.columns:
    data = recipes_df['n_ingredients'].dropna()
    mean_val = data.mean()
    std_val = data.std()

    # Count frequency
    counts = data.value_counts().sort_index()
    df_hist = pd.DataFrame({'n_ingredients': counts.index, 'count': counts.values})

    # Determine color based on statistical ranges
    def color_range(val):
        if val < mean_val - std_val:
            return 'red'
        elif val < mean_val:
            return 'blue'
        elif val < mean_val + std_val:
            return 'green'
        else:
            return 'orange'

    df_hist['color'] = df_hist['n_ingredients'].apply(color_range)




# In[13]:


plt.figure(figsize=(10, 6))
sns.histplot(data=recipes_df, x='n_ingredients', bins=range(1, 31), kde=False, color='skyblue')
plt.axvline(recipes_df['n_ingredients'].mean(), color='black', linestyle='--', label='Mean')
plt.axvline(recipes_df['n_ingredients'].mean() + recipes_df['n_ingredients'].std(), color='orange', linestyle=':', label='+1 STD')
plt.axvline(recipes_df['n_ingredients'].mean() - recipes_df['n_ingredients'].std(), color='red', linestyle=':', label='-1 STD')
plt.title('How Many Recipes Use X Ingredients?')
plt.xlabel('Number of Ingredients')
plt.ylabel('Number of Recipes')
plt.legend()
plt.tight_layout()
plt.show()


# In[14]:


plt.figure(figsize=(12, 4))
plt.plot(recipes_df['n_ingredients'].reset_index(drop=True), color='violet')
plt.axhline(recipes_df['n_ingredients'].mean(), color='black', linestyle='--', label='Mean')
plt.title('Ingredients Count per Recipe (Sequence)')
plt.xlabel('Recipe Index')
plt.ylabel('Number of Ingredients')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# ## Phase 4: Polishing the Silverware - Data Cleaning & Preprocessing
# 
# Raw data, like unwashed vegetables, needs cleaning before use. In this phase, we'll refine our datasets to make them more consistent and useful for Chefbelle.
# 
# Our key steps include:
# 1.  **Removing Duplicates:** Ensuring we don't have identical recipes cluttering the database.
# 2.  **Normalizing Ingredients:** Standardizing ingredient names (e.g., "Large Eggs" and "eggs" become "egg").
# 3.  **Handling Missing Values:** Addressing any gaps found during exploration (though our initial analysis showed minimal missing data in key fields).
# 4.  **Creating Dietary Tags:** Automatically identifying potential dietary characteristics (vegetarian, gluten-free, etc.) based on ingredients for easier filtering.

# ## 4.1. Tidying Up: Removing Duplicate Entries
# 
# This function helps us identify and remove duplicate recipes or interactions, keeping our data clean and efficient.

# In[15]:


# Function to check and remove duplicates in dataframes
def check_remove_duplicates(df, df_name, subset_cols=None):
    """
    Check and remove duplicates from a dataframe.

    Args:
        df: The dataframe to process
        df_name: Name of the dataframe for printing
        subset_cols: List of columns to consider for duplicates. If None, all columns are used.

    Returns:
        Dataframe with duplicates removed
    """
    print(f"\n{'-' * 30}")
    print(f"Duplicate analysis for {df_name}:")
    print(f"{'-' * 30}")

    # If subset not specified, identify potential key columns
    if subset_cols is None:
        # Try to find ID-like columns first
        id_cols = [col for col in df.columns if 'id' in col.lower()]
        name_cols = [col for col in df.columns if 'name' in col.lower()]

        if id_cols:
            subset_cols = id_cols
            print(f"Using ID columns for duplicate check: {subset_cols}")
        elif name_cols:
            subset_cols = name_cols
            print(f"Using name columns for duplicate check: {subset_cols}")
        else:
            # Use all columns if no suitable identifiers found
            subset_cols = df.columns.tolist()
            print("Using all columns for duplicate check")

    # Check for duplicates
    dup_count = df.duplicated(subset=subset_cols).sum()
    print(f"Number of duplicates in {df_name}: {dup_count} ({dup_count/len(df):.2%} of data)")

    if dup_count > 0:
        # Remove duplicates
        df_cleaned = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
        print(f"Number of records after removing duplicates: {len(df_cleaned)}")
        return df_cleaned
    else:
        print("No duplicates found")
        return df


# In[16]:


# Check and remove duplicates from all dataframes
print("\n=== DUPLICATE ANALYSIS FOR ALL DATAFRAMES ===")
recipes_df = check_remove_duplicates(recipes_df, "Recipes", subset_cols=['name'])
interactions_df = check_remove_duplicates(interactions_df, "Interactions")
nutrition_df = check_remove_duplicates(nutrition_df, "Nutrition")


# ## 4.2 Normalize Ingredient Names
# 
# Defines and applies a function `normalize_ingredients` to the `ingredients` column of `recipes_df`. This function converts ingredients to lowercase, removes leading quantities (simple regex), and strips whitespace. A new `normalized_ingredients` column is created. A sample comparison is printed.

# In[17]:


# Function to normalize ingredient names
def normalize_ingredients(ingredient_list):
    """
    Normalize ingredient names by removing quantities and standardizing format
    """
    normalized = []
    # If ingredient_list is already a list of strings
    if isinstance(ingredient_list, list):
        for ingredient in ingredient_list:
            # Skip empty ingredients
            if not ingredient or not isinstance(ingredient, str):
                continue

            # Remove quantities (simplified for demonstration)
            cleaned = re.sub(r'^\d+\s+\d+/\d+\s+', '', ingredient)
            cleaned = re.sub(r'^\d+/\d+\s+', '', cleaned)
            cleaned = re.sub(r'^\d+\s+', '', cleaned)

            # Convert to lowercase and strip whitespace
            cleaned = cleaned.lower().strip()

            normalized.append(cleaned)
    else:
        # Handle the case where ingredient_list might be a string or another format
        print("Warning: Expected ingredient_list to be a list, but got:", type(ingredient_list))
        if isinstance(ingredient_list, str):
            # Try to interpret as a string representation of a list
            try:
                actual_list = eval(ingredient_list) if ingredient_list.startswith('[') else [ingredient_list]
                return normalize_ingredients(actual_list)
            except:
                normalized = [ingredient_list.lower().strip()]

    return normalized

# Apply normalization to ingredients - with error handling
recipes_df['normalized_ingredients'] = recipes_df['ingredients'].apply(
    lambda x: normalize_ingredients(x) if isinstance(x, list) or isinstance(x, str) else []
)

# Show a sample recipe with normalized ingredients
if len(recipes_df) > 0:
    sample_idx = 0
    print(f"Original ingredients: {recipes_df.iloc[sample_idx]['ingredients']}")
    print(f"Normalized ingredients: {recipes_df.iloc[sample_idx]['normalized_ingredients']}")
else:
    print("No recipes found in the dataframe.")


# In[ ]:





# ## 4.3. Dietary Insights: Identifying Potential Tags 🏷️
# 
# - Can Chefbelle quickly find vegetarian or gluten-free options? This function performs basic keyword matching on the *normalized* ingredients to assign potential dietary tags. While not foolproof (complex recipes might need more advanced analysis), it provides a valuable starting point for filtering. We'll also visualize the prevalence of these tags.
# 
# - So we define a function `identify_dietary_tags` that assigns basic dietary tags (vegetarian, vegan, gluten-free, low-carb, dairy-free) based on keyword matching within the `normalized_ingredients`. Applies this function to create a `dietary_tags` column. It then generates a bar chart showing the distribution of these tags and prints sample recipes with their assigned tags.
# 
# 
# 

# In[18]:


# Function to identify dietary tags based on ingredients
def identify_dietary_tags(ingredients):
    """
    Identify dietary preferences based on ingredients
    """
    # Handle empty ingredients list
    if not ingredients or not isinstance(ingredients, (list, str)):
        return []

    # Convert list of ingredients to a single string for easier checking
    ingredients_str = ' '.join(ingredients).lower()

    tags = []

    # Vegetarian check (simplified)
    meat_ingredients = ['chicken', 'beef', 'pork', 'lamb', 'turkey', 'veal', 'bacon']
    if not any(meat in ingredients_str for meat in meat_ingredients):
        tags.append('vegetarian')

        # Vegan check (simplified)
        animal_products = ['cheese', 'milk', 'cream', 'yogurt', 'butter', 'egg', 'honey']
        if not any(product in ingredients_str for product in animal_products):
            tags.append('vegan')

    # Gluten-free check (simplified)
    gluten_ingredients = ['flour', 'wheat', 'barley', 'rye', 'pasta', 'bread']
    if not any(gluten in ingredients_str for gluten in gluten_ingredients):
        tags.append('gluten-free')

    # Low-carb check (simplified)
    high_carb_ingredients = ['sugar', 'pasta', 'rice', 'potato', 'bread', 'flour']
    if not any(carb in ingredients_str for carb in high_carb_ingredients):
        tags.append('low-carb')

    # Dairy-free check
    dairy_ingredients = ['milk', 'cheese', 'cream', 'yogurt', 'butter']
    if not any(dairy in ingredients_str for dairy in dairy_ingredients):
        tags.append('dairy-free')

    return tags

# Apply dietary tagging
recipes_df['dietary_tags'] = recipes_df['normalized_ingredients'].apply(identify_dietary_tags)

# Show the distribution of dietary tags
diet_counts = {}
for tags in recipes_df['dietary_tags']:
    for tag in tags:
        diet_counts[tag] = diet_counts.get(tag, 0) + 1

plt.figure(figsize=(10, 6))
diet_df = pd.Series(diet_counts).sort_values(ascending=False)
diet_df.plot(kind='bar')
plt.title('Distribution of Dietary Tags')
plt.xlabel('Dietary Tag')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Show sample recipes with their dietary tags
print("\nSample recipes with dietary tags:")
sample_recipes = recipes_df[['name', 'normalized_ingredients', 'dietary_tags']].sample(5)
for _, recipe in sample_recipes.iterrows():
    print(f"\nRecipe: {recipe['name']}")
    print(f"Ingredients: {', '.join(recipe['normalized_ingredients'])}")
    print(f"Dietary Tags: {', '.join(recipe['dietary_tags']) if recipe['dietary_tags'] else 'None'}")


# In[ ]:





# ## 4.4. Final Check: Processed Dataset Overview
# 
# - Let's quickly review the state of our datasets after cleaning and preprocessing. We'll check the number of records and the available columns in each DataFrame.
# - Prints a summary of the processed datasets, including the number of records and column names for `recipes_df`, `interactions_df`, and `nutrition_df`.

# In[19]:


# Basic dataset information
print("Raw Datasets information:")
print(f"Number of recipes: {len(recipes_df)}")
print("\nrecipes_df columns:")
print(recipes_df.columns.tolist())
print(15 * "-")
print(f"Number of interactions: {len(interactions_df)}")
print("\ninteractions_df columns:")
print(interactions_df.columns.tolist())
print(15 * "-")
print(f"Number of nutritions: {len(nutrition_df)}")
print("\nnutrition_df columns:")
print(nutrition_df.columns.tolist())
print(15 * "-")


# In[ ]:





# In[ ]:





# ## Phase 5: Building Chefbelle's Memory - Database Setup
# 
# With our data cleaned and prepped, it's time to build the databases that Chefbelle will rely on. We'll create two distinct storage systems:
# 
# 1.  **SQLite Database (`kitchen_db.sqlite`):** A relational database perfect for storing structured data like recipe details (ID, name, steps, contributor), user interactions (ratings, reviews), and potentially pre-calculated nutrition. It allows for precise lookups using SQL queries.
# 2.  **ChromaDB (Vector Database):** This database stores *embeddings* – numerical representations of recipe text (like names, ingredients, descriptions). It enables powerful *semantic search*, allowing Chefbelle to find recipes based on meaning and similarity ("find me something like a hearty beef stew," "show me cozy fall recipes") rather than just exact keywords.
# 
# This dual-database approach gives Chefbelle both structured lookup capabilities and flexible semantic understanding.

# In[20]:


import chromadb
print(chromadb.__version__)


# ## 5.1. Constructing the Databases: Setup Functions
# 
# These Python functions handle the creation and population of our SQLite and ChromaDB databases.
# 
# *   `setup_sql_database`: Takes our Pandas DataFrames and writes them into tables within the SQLite file. It includes preprocessing to ensure data types are compatible with SQL.
# *   `setup_vector_database`: Initializes ChromaDB, creates 'collections' (like tables) for recipes and interactions, processes the text data from our DataFrames to create documents and metadata, and adds these (along with their embeddings, which ChromaDB generates by default) to the vector store. This function uses the *pre-vectorized* data loaded from Kaggle input, saving significant processing time.

# In[21]:


# Define paths for SQL database
DB_PATH = "final/kitchen_db.sqlite"

#####################
# SQL Database Setup
#####################
def safe_convert(x):
    """
    Safely converts a value to a string:
      - If x is a list or numpy array, join its elements into a space-separated string.
      - If x is not a list/array and is not null, convert to string.
      - Otherwise, return an empty string.
    """
    if isinstance(x, (list, np.ndarray)):
        return " ".join([str(item) for item in x])
    return str(x) if pd.notna(x) else ""


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess DataFrame columns to be SQLite-compatible.
    """
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].apply(safe_convert)
    return df

def setup_sql_database(
    recipes_df: pd.DataFrame, 
    interactions_df: pd.DataFrame, 
    nutrition_df: Optional[pd.DataFrame] = None,
    db_path: str = DB_PATH
) -> sqlite3.Connection:
    """
    Set up SQLite database with raw dataframes.
    """
    recipes_df = preprocess_dataframe(recipes_df)
    interactions_df = preprocess_dataframe(interactions_df)
    if nutrition_df is not None:
        nutrition_df = preprocess_dataframe(nutrition_df)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    print(f"Creating SQLite database at {db_path}")
    conn = sqlite3.connect(db_path)

    print(f"Storing {len(recipes_df)} recipes in the database")
    recipes_df.to_sql('recipes', conn, if_exists='replace', index=False)
    print(f"Storing {len(interactions_df)} interactions in the database")
    interactions_df.to_sql('interactions', conn, if_exists='replace', index=False)

    if nutrition_df is not None:
        print(f"Storing {len(nutrition_df)} nutrition entries in the database")
        nutrition_df.to_sql('nutrition', conn, if_exists='replace', index=False)

    print("SQL database setup complete")
    return conn

#############################
# Vector Database Setup (ChromaDB)
#############################
def setup_vector_database(
    vectorized_recipes_df: pd.DataFrame,
    vectorized_interactions_df: Optional[pd.DataFrame] = None,
    vector_db_path: str = VECTOR_DB_PATH
) -> Tuple[Any, Any, Optional[Any]]:
    """
    Set up ChromaDB using the precomputed dataframes for recipes and interactions.

    Arguments:
        vectorized_recipes_df: DataFrame with your recipe data.
        vectorized_interactions_df: DataFrame with your interaction data.
        vector_db_path: Directory where ChromaDB will store its data.

    Returns:
        A tuple containing the ChromaDB client, the recipe collection, and 
        the interactions collection (if interactions_df is provided).
    """
    os.makedirs(vector_db_path, exist_ok=True)
    print(f"Creating ChromaDB at {vector_db_path}")
    client = chromadb.PersistentClient(path=vector_db_path)

    #########################
    # Load recipes into ChromaDB
    #########################
    print(f"Setting up recipe collection with {len(vectorized_recipes_df)} recipes")
    recipe_collection = client.get_or_create_collection(name="recipes")

    recipe_documents = []
    recipe_metadatas = []
    recipe_ids = []

    # Define which recipe columns to include as metadata
    metadata_fields = ['name', 'minutes', 'contributor_id', 'submitted',
                       'tags', 'nutrition', 'n_steps', 'cuisine_type',
                       'n_ingredients', 'dietary_tags']

    for i, row in vectorized_recipes_df.iterrows():
        # Determine a unique recipe ID. Use 'id' column if available.
        recipe_id = row.get('id')
        if recipe_id is None or (isinstance(recipe_id, float) and pd.isna(recipe_id)) or recipe_id == "":
            recipe_id = str(i)
        else:
            recipe_id = str(recipe_id)
        recipe_ids.append(recipe_id)

        # Build a document string by concatenating key text fields.
        # You may adjust the fields below to better capture recipe information.
        doc_text = " ".join([
            safe_convert(row.get('name', '')),
            safe_convert(row.get('ingredients', '')),
            safe_convert(row.get('steps', '')),
            safe_convert(row.get('description', ''))
        ])
        recipe_documents.append(doc_text)

        # Build richer metadata from the chosen fields.
        metadata = {key: safe_convert(row.get(key, "")) for key in metadata_fields}
        metadata['recipe_id'] = recipe_id
        recipe_metadatas.append(metadata)

    batch_size = 1000
    for j in range(0, len(recipe_documents), batch_size):
        end_idx = min(j + batch_size, len(recipe_documents))
        recipe_collection.add(
            documents=recipe_documents[j:end_idx],
            metadatas=recipe_metadatas[j:end_idx],
            ids=recipe_ids[j:end_idx]
        )

    #########################
    # Load interactions into ChromaDB (if provided)
    #########################
    interactions_collection = None
    if vectorized_interactions_df is not None and not vectorized_interactions_df.empty:
        print(f"Setting up interactions collection with {len(vectorized_interactions_df)} interactions")
        interactions_collection = client.get_or_create_collection(name="interactions")

        interaction_documents = []
        interaction_metadatas = []
        interaction_ids = []

        for i, row in vectorized_interactions_df.iterrows():
            # Create a unique interaction ID from user_id, recipe_id, and index.
            user_id = safe_convert(row.get('user_id', ''))
            recipe_id = safe_convert(row.get('recipe_id', ''))
            interaction_id = f"{user_id}_{recipe_id}_{i}"
            interaction_ids.append(interaction_id)

            # Use the review text as the primary document.
            review_text = safe_convert(row.get('review', ''))
            if not review_text:
                review_text = "No review provided."
            interaction_documents.append(review_text)

            # Build metadata for this interaction.
            int_metadata = {
                'interaction_id': interaction_id,
                'user_id': user_id,
                'recipe_id': recipe_id,
                'date': safe_convert(row.get('date', '')),
                'rating': safe_convert(row.get('rating', ''))
            }
            interaction_metadatas.append(int_metadata)

        for j in range(0, len(interaction_documents), batch_size):
            end_idx = min(j + batch_size, len(interaction_documents))
            interactions_collection.add(
                documents=interaction_documents[j:end_idx],
                metadatas=interaction_metadatas[j:end_idx],
                ids=interaction_ids[j:end_idx]
            )

    print("Vector database setup complete")
    return client, recipe_collection, interactions_collection




# ## 5.2. Populating the Databases (Execution)
# 
# This block executes the database setup functions defined above.
# 
# *   It calls `setup_sql_database` to create/replace the `kitchen_db.sqlite` file with the latest processed data from our DataFrames (`recipes_df`, `interactions_df`, `nutrition_df`).
# *   The call to `setup_vector_database` is **commented out** because we are using a *pre-existing* vector database loaded from Kaggle input (`/kaggle/input/food-com-vectorized-with-chromadb/vector_db`). Creating embeddings for hundreds of thousands of recipes takes considerable time, so we've prepared it beforehand. If you were running this from scratch, you would uncomment this call.
# 
# After this step, our SQLite database is ready for structured queries, and our ChromaDB vector database is ready for semantic searches.

# In[22]:


##############################
# Main Execution
##############################
if __name__ == "__main__":
    # Assume recipes_df and interactions_df have been loaded previously.
    # For example:
    # recipes_df = pd.read_pickle("your_recipes.pkl")
    # interactions_df = pd.read_pickle("your_interactions.pkl")

    # Set up the SQL database
    sqlite_conn = setup_sql_database(
        recipes_df=recipes_df,
        interactions_df=interactions_df,
        nutrition_df=nutrition_df,  # Modify if you have nutrition data.
        db_path=DB_PATH
    )

    # Set up ChromaDB with recipes and interactions
    # chroma_client, recipe_collection, interactions_collection = setup_vector_database(
    #     vectorized_recipes_df=recipes_df,
    #     vectorized_interactions_df=interactions_df,
    #     vector_db_path=VECTOR_DB_PATH
    # )

    print("ChromaDB is ready for similarity search!")


# In[ ]:





# ## Phase 6: Checking the Pantry - Database Verification
# 
# Before Chefbelle starts cooking, let's peek inside her newly organized pantry (our databases) to make sure everything is in order.
# 
# These utility functions help us inspect the ChromaDB vector database:
# *   `view_schema_info`: Examines a sample of records in a ChromaDB collection to show us the available metadata fields (like `name`, `minutes`, `cuisine_type`) and their likely data types.
# *   `collection_info`: Lists all collections within the ChromaDB instance and provides a quick summary (name, record count, sample IDs).

# In[23]:


# Path to SQL database
DB_PATH = "/kaggle/working/final/kitchen_db.sqlite"
# Path to Vectorized database
VECTOR_DB_PATH = "/kaggle/input/food-com-vectorized-with-chromadb/vector_db"


def view_schema_info(collection_name: str, db_path: str = VECTOR_DB_PATH):
    """
    View schema information for a collection (metadata fields and their data types).

    Args:
        collection_name: Name of the collection to analyze
        db_path: Path to the ChromaDB database
    """
    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection(name=collection_name)
    except ValueError as e:
        print(f"Collection '{collection_name}' not found. Error: {str(e)}")
        return None

    # Get a sample of records to analyze schema
    try:
        results = collection.get(
            limit=100,
            include=['metadatas']
        )

        if not results['metadatas']:
            print(f"Collection '{collection_name}' is empty or has no metadata.")
            return None

        # Analyze metadata fields
        print(f"\n=== Schema for '{collection_name}' collection ===\n")
        print("Metadata fields:")

        # Collect all possible keys and their types
        all_keys = set()
        key_types = {}
        key_examples = {}

        for metadata in results['metadatas']:
            for key, value in metadata.items():
                all_keys.add(key)

                # Track the data type
                value_type = type(value).__name__
                if key not in key_types:
                    key_types[key] = set()
                key_types[key].add(value_type)

                # Store an example value
                if key not in key_examples and value:
                    example = str(value)
                    if len(example) > 50:
                        example = example[:50] + "..."
                    key_examples[key] = example

        # Display the schema information
        for key in sorted(all_keys):
            types_str = ", ".join(key_types[key])
            example = key_examples.get(key, "N/A")
            print(f"  - {key}: {types_str}")
            print(f"    Example: {example}")

        return key_types

    except Exception as e:
        print(f"Error getting schema info: {str(e)}")
        return None


def collection_info(db_path: str = VECTOR_DB_PATH):
    """
    A simple function to display basic information about all collections.
    More robust against API changes than the other functions.

    Args:
        db_path: Path to the ChromaDB database
    """
    client = chromadb.PersistentClient(path=db_path)

    try:
        collection_names = client.list_collections()
        print(f"Found {len(collection_names)} collections in {db_path}:")

        for name in collection_names:
            print(f"\nCollection: {name}")

            try:
                collection = client.get_collection(name=str(name))

                # Try to get count
                try:
                    count = collection.count(where={})
                    print(f"  Records: {count}")
                except:
                    print("  Count: Could not retrieve")

                # Try to get the first few items
                try:
                    first_items = collection.get(limit=3, include=["metadatas"])
                    print(f"  Sample IDs: {first_items['ids']}")

                    # Show first item metadata as example
                    if first_items['metadatas'] and len(first_items['metadatas']) > 0:
                        print("  Sample metadata keys:", list(first_items['metadatas'][0].keys()))
                except:
                    print("  Sample: Could not retrieve")

            except Exception as e:
                print(f"  Error accessing collection: {str(e)}")

    except Exception as e:
        print(f"Error listing collections: {str(e)}")


# ## 6.1. Listing ChromaDB Collections
# 
# Let's connect to our pre-built ChromaDB vector database and see what collections are inside. We expect to see `recipes` and `interactions`.

# In[24]:


client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
print(client.list_collections())


# In[ ]:





# ## 6.2. ChromaDB Collection Summaries
# 
# Using our helper function, let's get a quick summary of each collection, including the number of items and sample metadata.

# In[25]:


collection_info(VECTOR_DB_PATH)


# In[ ]:





# ## 6.3. Inspecting the 'recipes' Collection Schema
# 
# Let's dive deeper into the `recipes` collection. What specific metadata fields are available for filtering and display?

# In[26]:


view_schema_info("recipes", VECTOR_DB_PATH)


# In[ ]:





# ## 6.4. Inspecting the 'interactions' Collection Schema
# 
# Now, let's examine the `interactions` collection to see the metadata associated with user reviews and ratings.

# In[27]:


view_schema_info("interactions", VECTOR_DB_PATH)


# In[ ]:





# ## 6.5. Enabling Smart Search: Similarity Functions
# 
# With ChromaDB verified, we can define functions that leverage its power for semantic search. These functions allow Chefbelle to find relevant recipes or reviews based on natural language queries.
# 
# *   `gemini_recipe_similarity_search`: Searches the `recipes` collection. Takes a query text and optional filters (cuisine, dietary tags, max time) and returns a list of the most similar recipes with their metadata and similarity scores.
# *   `gemini_interaction_similarity_search`: Searches the `interactions` collection (primarily the review text) to find reviews similar to the query text.

# In[28]:


def gemini_recipe_similarity_search(query_text: str, n_results: int, cuisine: Optional[str] = None, dietary_tag: Optional[str] = None, max_minutes: Optional[int] = None) -> str:
    """
    Searches for similar recipes based on a query, with optional filters and returns full metadata.

    Args:
        query_text: The text to search for in recipes.
        n_results: The number of top similar recipes to return.
        cuisine: (Optional) Filter by cuisine type (e.g., 'mexican', 'italian').
        dietary_tag: (Optional) Filter by dietary tag (e.g., 'vegetarian', 'gluten-free').
        max_minutes: (Optional) Filter recipes with a cooking time less than or equal to this value.

    Returns:
        A formatted string containing the full metadata of the top similar recipes with similarity scores.
    """
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        recipe_collection = client.get_collection(name="recipes")

        where_clause = {}
        if cuisine is not None:
            where_clause["cuisine_type"] = cuisine
        if dietary_tag is not None:
            where_clause["dietary_tags"] = {"$contains": dietary_tag}
        if max_minutes is not None:
            where_clause["minutes"] = {"$lte": str(max_minutes)} # Store as string in metadata

        results = recipe_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        if not results['ids'][0]:
            return f"No similar recipes found for the query: '{query_text}' with the specified criteria."

        output = f"Found {len(results['ids'][0])} similar recipes for query: '{query_text}'.\n"
        output += "-" * 80 + "\n"
        for i, (doc_id, doc, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            similarity_score = (1 - distance) * 100
            output += f"\n{i+1}. Recipe Name: {metadata.get('name', 'Unnamed')}\n"
            output += f"   Similarity: {similarity_score:.2f}%\n"
            output += f"   Recipe ID: {doc_id}\n"
            for key, value in metadata.items():
                output += f"   {key.replace('_', ' ').title()}: {value}\n"
            output += f"   Ingredients: {doc}\n"  # Include the full document (ingredients/steps)
            output += "-" * 80 + "\n"

        return output

    except Exception as e:
        return f"Error during recipe similarity search: {e}"

# Updated `gemini_interaction_similarity_search` Function:

def gemini_interaction_similarity_search(query_text: str, n_results: int) -> str:
    """
    Searches for similar user interactions (reviews) based on a query and returns full metadata.

    Args:
        query_text: The text to search for in user reviews.
        n_results: The number of top similar interactions to return.

    Returns:
        A formatted string containing the full metadata of the top similar interactions with similarity scores.
    """
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        interactions_collection = client.get_collection(name="interactions")
        results = interactions_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        if not results['ids'][0]:
            return f"No similar reviews found for the query: '{query_text}'."

        output = f"Found {len(results['ids'][0])} similar reviews for query: '{query_text}'.\n"
        output += "-" * 80 + "\n"
        for i, (doc_id, doc, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            similarity_score = (1 - distance) * 100
            output += f"\n{i+1}. Review ID: {doc_id}\n"
            output += f"   Similarity: {similarity_score:.2f}%\n"
            for key, value in metadata.items():
                output += f"   {key.replace('_', ' ').title()}: {value}\n"
            output += f"   Review Text: {doc}\n"  # Include the full document (review text)
            output += "-" * 80 + "\n"

        return output

    except ValueError:
        return "Interactions collection not found. Make sure you have interaction data loaded."
    except Exception as e:
        return f"Error during interaction similarity search: {e}"


# ## 6.6. Testing Recipe Search: Finding Italian Pizza 🍕
# 
# Let's test our recipe search! We'll ask for recipes related to "Italian pizza" and see the top match returned by ChromaDB's semantic search.

# In[29]:


query_text = "check for making an italian pizza "
result = gemini_recipe_similarity_search(query_text, n_results = 1)
print(result)


# ## 6.7. Testing Interaction Search: Finding Pizza Reviews
# 
# Now, let's search the *reviews* for mentions of "best Italian pizza" to test the interaction search.

# In[30]:


query_text = "best italian pizza"
result = gemini_interaction_similarity_search(query_text, n_results = 1)
print(result)


# ## Phase 7: Equipping Chefbelle - Defining Agent Tools 🛠️
# 
# Now we start building Chefbelle's core capabilities. An "Agent" in frameworks like LangGraph relies on "Tools" – specific functions it can call to perform actions or retrieve information.
# 
# Here, we define the Python functions that will serve as Chefbelle's tools:
# 
# *   **Database Tools:**
#     *   `list_tables`: See what tables are in the SQLite DB.
#     *   `describe_table`: Understand the structure (columns) of a specific table.
#     *   `execute_query`: Run SQL queries to fetch specific data (read-only for safety).
#     *   `get_recipe_by_id`: Retrieve all details for a specific recipe ID from SQLite.
#     *   `get_ratings_and_reviews_by_recipe_id`: Fetch ratings and reviews for a recipe from SQLite.
# *   **External API Tools:**
#     *   `fetch_nutrition_from_usda_fdc`: Get detailed nutritional information for a *single* ingredient from the official USDA FoodData Central API (requires API key). This provides reliable, standardized data.
#     *   `fetch_live_recipe_data` (Placeholder): Simulates fetching the *latest* version of a recipe directly from food.com (would involve web scraping in a full implementation).
# *   **Custom Logic Tools:**
#     *   `customize_recipe` (Placeholder): Simulates modifying a recipe based on user requests (e.g., "make it vegan"). This would involve more complex LLM logic in a full version.
# 
# These tools give Chefbelle the ability to access structured data, query external knowledge bases, and perform specialized tasks.

# In[31]:


# Assume DB_PATH is defined elsewhere
# DB_PATH = "your_database_path.db"

# --- Database Functions (Mostly unchanged, added try/finally and context managers) ---

def list_tables() -> List[str]:
    """List all tables in the SQLite database using context managers."""
    tables = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"Error listing tables: {e}")
        # Depending on desired behavior, you might return [] or raise e
    return tables

def describe_table(table_name: str) -> List[Tuple[str, str]]:
    """Describe the schema of a specified table using context managers."""
    schema = []
    try:
        # Basic validation/sanitization - prevent SQL injection in table names
        if not table_name.isalnum() and '_' not in table_name:
             print(f"Warning: Invalid table name format '{table_name}'. Skipping.")
             return []
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Use parameterized query even for PRAGMA if possible, or ensure table_name is safe
            # For PRAGMA table_info, direct insertion is common but requires validation ^
            cursor.execute(f"PRAGMA table_info({table_name});")
            schema_raw = cursor.fetchall()
            # Extract relevant columns (name, type) - indices 1 and 2
            schema = [(col[1], col[2]) for col in schema_raw]
    except sqlite3.Error as e:
        print(f"Error describing table '{table_name}': {e}")
    return schema


def execute_query(sql: str) -> List[Tuple]:
    """Execute a potentially read-only SQL query and return the results using context managers."""
    results = []
    # Basic check to prevent obviously harmful commands - enhance as needed
    if not sql.strip().upper().startswith("SELECT") and not sql.strip().upper().startswith("PRAGMA"):
         print("Warning: Only SELECT and PRAGMA queries are recommended via execute_query.")
         # return [("Error:", "Potentially unsafe query blocked.")] # Or allow if you trust the source
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Error executing SQL query: {e}")
        # Return error message structured like results for consistency
        return [("Error executing SQL query:", str(e))]
    return results

# --- Modified Open Food Facts Function with Retries ---

def fetch_nutrition_from_openfoodfacts(ingredient_name: str) -> Dict:
    """
    Fetch nutrition data for an ingredient from Open Food Facts API.
    Includes retry logic for rate limiting (429) and transient errors.
    """
    #api_key = os.getenv('OPENFOODFACTS_API_KEY')
    # You might still want a warning if the key isn't set, but OFF search often works without it.

    search_url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "search_terms": ingredient_name,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 1 # We only need the top result
    }
    headers = {'User-Agent': 'CapstoneProject/1.0 (Language Model Integration)'} # Good practice

    max_retries = 3 # Internal retry limit
    base_timeout = 15 # Internal timeout
    retry_delay = 1 # Initial delay in seconds for retries

    for attempt in range(max_retries):
        try:
            response = requests.get(search_url, params=params, headers=headers, timeout=base_timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            data = response.json()

            if data.get('products') and len(data['products']) > 0:
                product = data['products'][0]
                nutriments = product.get('nutriments', {})

                nutrition_info = {
                    "food_normalized": ingredient_name,
                    "source": "Open Food Facts",
                    "product_name": product.get('product_name', 'N/A'),
                    "calories_100g": nutriments.get('energy-kcal_100g'),
                    "fat_100g": nutriments.get('fat_100g'),
                    "saturated_fat_100g": nutriments.get('saturated-fat_100g'),
                    "carbohydrates_100g": nutriments.get('carbohydrates_100g'),
                    "sugars_100g": nutriments.get('sugars_100g'),
                    "fiber_100g": nutriments.get('fiber_100g'),
                    "proteins_100g": nutriments.get('proteins_100g'),
                    "sodium_100g": nutriments.get('sodium_100g'),
                }
                return {k: v for k, v in nutrition_info.items() if v is not None}
            else:
                # No product found, not an error, just unavailable
                return {"status": "unavailable", "reason": f"No product found for '{ingredient_name}' on Open Food Facts"}

        except requests.exceptions.HTTPError as e:
            # Specific handling for Rate Limiting (429)
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (retry_delay * (2 ** attempt)) + random.uniform(0, 1)
                    print(f"Rate limit hit for '{ingredient_name}'. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue # Retry the loop
                else:
                    print(f"Rate limit hit for '{ingredient_name}'. Max retries exceeded.")
                    return {"status": "unavailable", "reason": f"API rate limit exceeded after {max_retries} attempts: {e}"}
            # Handle other HTTP errors (e.g., 5xx server errors) potentially with retries too
            elif e.response.status_code >= 500 and attempt < max_retries - 1:
                 wait_time = (retry_delay * (2 ** attempt)) + random.uniform(0, 1)
                 print(f"Server error ({e.response.status_code}) for '{ingredient_name}'. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                 time.sleep(wait_time)
                 continue # Retry the loop
            else:
                # For other client errors (4xx) or server errors after retries, report failure
                print(f"HTTP Error fetching nutrition for '{ingredient_name}': {e}")
                return {"status": "unavailable", "reason": f"API request failed with HTTP error: {e}"}

        except requests.exceptions.RequestException as e:
            # Handle other connection/timeout errors
            if attempt < max_retries - 1:
                wait_time = (retry_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"Request error for '{ingredient_name}': {e}. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue # Retry the loop
            else:
                print(f"Error fetching nutrition for '{ingredient_name}' after {max_retries} attempts: {e}")
                return {"status": "unavailable", "reason": f"API request failed after retries: {e}"}

        except json.JSONDecodeError:
            # If response is not valid JSON
            print(f"Error decoding JSON response for '{ingredient_name}'")
            # No retry for decoding error usually, indicates bad response content
            return {"status": "unavailable", "reason": "Invalid JSON response from API"}

    # Should not be reached if loop completes, but as a fallback:
    return {"status": "unavailable", "reason": "Max retries exceeded without success"}


# --- Modified Recipe Function with Small Delay ---

def get_recipe_by_id(recipe_id: str) -> Optional[dict]:
    """Get a recipe by its ID, including live nutrition data (with delays & retries)."""
    recipe = None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row # Return rows that act like dictionaries
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM recipes WHERE id = ?", (recipe_id,))
            recipe_data = cursor.fetchone()

            if not recipe_data:
                return None

            # Convert Row object to a mutable dictionary
            recipe = dict(recipe_data)

            # --- Field Parsing Logic (Keep your existing logic, maybe add more logging) ---
            for field in ["steps", "ingredients", "nutrition", "tags", "dietary_tags", "normalized_ingredients"]:
                value = recipe.get(field)
                if isinstance(value, str):
                    try:
                        recipe[field] = json.loads(value)
                        # print(f"Successfully parsed JSON for field '{field}' in recipe ID {recipe_id}.") # Optional: Success log
                    except json.JSONDecodeError:
                        if field in ["ingredients", "tags", "dietary_tags", "normalized_ingredients"]:
                            # Fallback: Split potentially space or comma-separated strings
                            # Consider more robust splitting if needed (e.g., handle commas)
                            potential_list = [item.strip() for item in value.replace(',', ' ').split() if item.strip()]
                            if potential_list:
                                recipe[field] = potential_list
                                print(f"Info: Field '{field}' in recipe ID {recipe_id} treated as separated string -> {recipe[field]}")
                            else:
                                print(f"Warning: Field '{field}' in recipe ID {recipe_id} was string but empty after split.")
                                recipe[field] = [] # Ensure it's an empty list
                        elif field == "steps":
                            print(f"Warning: Could not parse JSON for field 'steps' in recipe ID {recipe_id}. Kept as string.")
                            # Keep as string is fine here
                        else: # E.g., nutrition field if not JSON
                            print(f"Warning: Could not parse JSON for field '{field}' in recipe ID {recipe_id}. Value: {value[:100]}...")
                            # Decide how to handle - keep string, set to None, etc.
                            pass
                # Ensure expected list fields are indeed lists if they exist but aren't strings
                elif field in ["ingredients", "tags", "dietary_tags", "normalized_ingredients"] and value is not None and not isinstance(value, list):
                     print(f"Warning: Field '{field}' in recipe ID {recipe_id} was type {type(value)}, expected list or string. Attempting conversion.")
                     try:
                         recipe[field] = list(value) # Basic conversion attempt
                     except TypeError:
                         print(f"Error: Could not convert field '{field}' to list for recipe ID {recipe_id}. Setting to empty list.")
                         recipe[field] = []


            # --- Fetch nutrition for normalized ingredients ---
            ingredient_nutrition_list = []
            normalized_ingredients = recipe.get("normalized_ingredients")

            if isinstance(normalized_ingredients, list):
                for i, ingredient in enumerate(normalized_ingredients):
                    if isinstance(ingredient, str) and ingredient.strip():
                        print(f"Fetching nutrition for: '{ingredient}' (Item {i+1}/{len(normalized_ingredients)})") # Log progress
                        nutrition_data = fetch_nutrition_from_usda_fdc(ingredient)
                        ingredient_nutrition_list.append(nutrition_data)
                        # *** ADD A SMALL DELAY HERE ***
                        time.sleep(random.uniform(0.5, 1.5)) # Wait 0.5 to 1.5 seconds before the next call
                    elif not isinstance(ingredient, str):
                         print(f"Warning: Skipping non-string item in normalized_ingredients: {ingredient}")
                         ingredient_nutrition_list.append({"status": "skipped", "reason": f"Invalid ingredient format: {type(ingredient)}"})
                    # else: skip empty strings silently
            elif normalized_ingredients is not None:
                print(f"Error: 'normalized_ingredients' field in recipe ID {recipe_id} is not a list after processing. Type: {type(normalized_ingredients)}")
                ingredient_nutrition_list.append({"status": "error", "reason": "normalized_ingredients field could not be processed into a list"})

            recipe['ingredient_nutrition'] = ingredient_nutrition_list

    except sqlite3.Error as e:
        print(f"Database error getting recipe ID {recipe_id}: {e}")
        return None # Or raise the error if preferred
    except Exception as e: # Catch other potential errors
        print(f"An unexpected error occurred in get_recipe_by_id for {recipe_id}: {e}")
        # Return the partially processed recipe if available, or None
        return recipe if recipe else None

    return recipe


# --- Ratings Function (Using context manager) ---
def get_ratings_and_reviews_by_recipe_id(recipe_id: str, limit: int) -> Optional[dict]:
    """Get ratings and recent reviews for a recipe ID using context managers."""
    overall_rating = None
    reviews_list = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Get overall rating
            cursor.execute("SELECT AVG(rating) FROM interactions WHERE recipe_id = ?", (recipe_id,))
            overall_rating_result = cursor.fetchone()
            # Ensure we handle None if no ratings exist before accessing index 0
            if overall_rating_result and overall_rating_result[0] is not None:
                 overall_rating = round(overall_rating_result[0], 2) # Round for cleaner display


            # Get most recent reviews
            cursor.execute(
                "SELECT date, rating, review FROM interactions WHERE recipe_id = ? AND review IS NOT NULL AND review != '' ORDER BY date DESC LIMIT ?",
                (recipe_id, limit),
            )
            recent_reviews = cursor.fetchall()
            columns = ["date", "rating", "review"]
            reviews_list = [dict(zip(columns, review)) for review in recent_reviews]

    except sqlite3.Error as e:
        print(f"Database error getting ratings/reviews for recipe ID {recipe_id}: {e}")
        # Return partial data or indicate error
        return {"overall_rating": overall_rating, "recent_reviews": [], "error": str(e)}

    return {"overall_rating": overall_rating, "recent_reviews": reviews_list}


# ## 7.1. Testing an Alternative Nutrition API (Open Food Facts - Commented Out)
# 
# Initially, we might explore different nutrition APIs. This cell shows a test call to Open Food Facts. However, we found the USDA FDC API to be more consistently reliable for standardized ingredient data, so we'll proceed with that. This call remains commented out.

# In[32]:


fetch_nutrition_from_openfoodfacts("apple")


# ## 7.2. Defining the USDA Nutrition Tool
# 
# This function (`fetch_nutrition_from_usda_fdc`) is a key tool for Chefbelle. It takes an ingredient name, queries the comprehensive USDA FoodData Central database via its API, and returns detailed nutritional information (calories, macros, etc.) per 100g.
# 
# *   **Reliability:** Uses the official USDA database.
# *   **Robustness:** Includes automatic retries for common API issues (like rate limits).
# *   **Mapping:** Translates nutrient names from the API into a consistent format for Chefbelle.
# 
# We'll test it by fetching data for "raw apple."

# In[33]:


# Consider getting the API key from an environment variable for security
# Example: export USDA_API_KEY='USDA_API_KEY'
# If using an environment variable:
# USDA_API_KEY = os.environ.get("USDA_API_KEY")
# Or pass it directly as an argument to the function.

# Mapping from FDC nutrient names (or IDs for more stability) to our desired keys.
# Using names here for readability. Units are typically per 100g in FDC.
# Note: FDC uses 'KCAL' for calories, 'G' for macros, 'MG' for sodium.
FDC_NUTRIENT_MAP = {
    # Nutrient Name in FDC API : Target Key
    "Energy": "calories_100g", # Often unit KCAL
    "Total lipid (fat)": "fat_100g", # Often unit G
    "Fatty acids, total saturated": "saturated_fat_100g", # Often unit G
    "Carbohydrate, by difference": "carbohydrates_100g", # Often unit G
    "Sugars, total including NLEA": "sugars_100g", # Often unit G
    "Fiber, total dietary": "fiber_100g", # Often unit G
    "Protein": "proteins_100g", # Often unit G
    "Sodium, Na": "sodium_100g", # Often unit MG
}



# @tool # Uncomment this if you are using it as a LangChain/LangGraph tool
def fetch_nutrition_from_usda_fdc(ingredient_name: str) -> str:
    """
    Fetches nutrition data (per 100g) for a single ingredient from USDA FoodData Central API.
    Requires a USDA FDC API key. Includes robust retry logic.
    Returns nutrition data as a JSON string or an error/unavailable status.
    """
    print(f"DEBUG TOOL CALL: fetch_nutrition_from_usda_fdc(ingredient_name='{ingredient_name}')")
    api_key = UserSecretsClient().get_secret("USDA_API_KEY")

    if not api_key:
        print("ERROR: USDA FDC API key is required.")
        return json.dumps({"error": "USDA FDC API key was not provided."})

    search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": ingredient_name,
        "api_key": api_key,
        "pageSize": 1, # Get the top hit
        "dataType": "SR Legacy,Foundation", # Prioritize standard reference / foundation foods for generic ingredients
        # Consider adding "Branded" if you need specific packaged products
    }
    headers = {'User-Agent': 'KitchenAssistantLangGraph/1.0 (Language: Python)'} # Good practice

    max_retries = 3
    base_timeout = 15
    retry_delay = 1 # Initial delay

    for attempt in range(max_retries): # 0, 1, 2 (3 attempts total)
        try:
            response = requests.get(search_url, params=params, headers=headers, timeout=base_timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if data.get('foods') and len(data['foods']) > 0:
                food_item = data['foods'][0]
                fdc_nutrients = food_item.get('foodNutrients', [])

                # Extract desired fields using the mapping
                nutrition_info = {
                    "food_normalized": ingredient_name,
                    "source": "USDA FoodData Central",
                    "product_name": food_item.get('description', ingredient_name), # Use FDC description
                    "fdc_id": food_item.get('fdcId'), # Useful identifier
                    "data_type": food_item.get('dataType'), # e.g., SR Legacy, Branded
                }

                # Iterate through nutrients reported by FDC for this food
                found_nutrients = {}
                for nutrient in fdc_nutrients:
                    nutrient_name = nutrient.get('nutrientName')
                    nutrient_unit = nutrient.get('unitName')
                    nutrient_value = nutrient.get('value')

                    # Check if this nutrient is one we want to map
                    target_key = FDC_NUTRIENT_MAP.get(nutrient_name)
                    if target_key:
                         # Optional: Check if the unit matches expected (e.g., KCAL for Energy)
                         # expected_unit = FDC_EXPECTED_UNITS.get(target_key)
                         # if nutrient_unit == expected_unit:
                        found_nutrients[target_key] = nutrient_value
                         # else:
                         #    print(f"Warning: Unit mismatch for {target_key}: Expected {expected_unit}, Got {nutrient_unit}")


                # Add found nutrients to the main dictionary
                nutrition_info.update(found_nutrients)

                # Filter out None values BEFORE checking core nutrients
                # (Note: FDC usually returns 0 rather than null/None for zero values)
                filtered_nutrition = {k: v for k, v in nutrition_info.items() if v is not None}

                # Check if at least one core nutrient is present and numeric
                core_nutrients = ["calories_100g", "fat_100g", "proteins_100g", "carbohydrates_100g"]
                has_core_data = False
                for core_key in core_nutrients:
                    if core_key in filtered_nutrition:
                        try:
                            # Check if it's actually a number (or can be converted)
                            float(filtered_nutrition[core_key])
                            has_core_data = True
                            break # Found at least one valid core nutrient
                        except (ValueError, TypeError):
                            continue # Skip if not numeric

                if not has_core_data:
                    print(f"--> No core numeric nutrition data found for '{ingredient_name}' in product '{filtered_nutrition.get('product_name', 'N/A')}' (FDC ID: {filtered_nutrition.get('fdc_id')})")
                    return json.dumps({"status": "unavailable", "reason": f"No detailed numeric core nutrition data found for '{ingredient_name}'"})

                # Success: return JSON string
                print(f"--> Successfully found nutrition data for '{ingredient_name}' via USDA FDC")
                return json.dumps(filtered_nutrition, indent=2)
            else:
                # No food found for the query
                print(f"--> No product found for '{ingredient_name}' via USDA FDC")
                # Try again with Branded data type? Or just report unavailable.
                # Let's report unavailable for now.
                return json.dumps({"status": "unavailable", "reason": f"No product found for '{ingredient_name}' on USDA FDC"})

        except requests.exceptions.HTTPError as e:
            # Specific handling for FDC API key errors (403 Forbidden often indicates bad key)
            if e.response.status_code == 403:
                 print(f"HTTP Error 403 (Forbidden) for '{ingredient_name}'. Check your USDA FDC API key.")
                 return json.dumps({"status": "error", "reason": f"API request failed with HTTP 403 (Forbidden). Check API Key."})
            elif e.response.status_code == 429 and attempt < max_retries - 1:
                wait_time = (retry_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"Rate limit hit for '{ingredient_name}'. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            elif e.response.status_code >= 500 and attempt < max_retries - 1:
                wait_time = (retry_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"Server error ({e.response.status_code}) for '{ingredient_name}'. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                print(f"HTTP Error fetching nutrition for '{ingredient_name}': {e}")
                return json.dumps({"status": "unavailable", "reason": f"API request failed with HTTP error: {e}"})

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (retry_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"Request error for '{ingredient_name}': {e}. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                print(f"Error fetching nutrition for '{ingredient_name}' after {max_retries} attempts: {e}")
                return json.dumps({"status": "unavailable", "reason": f"API request failed after retries: {e}"})

        except json.JSONDecodeError:
            print(f"Error decoding JSON response for '{ingredient_name}'")
            return json.dumps({"status": "unavailable", "reason": "Invalid JSON response from API"})

        except Exception as e:
             print(f"ERROR in fetch_nutrition_from_usda_fdc: {e}")
             import traceback
             traceback.print_exc()
             return json.dumps({"error": f"Unexpected error fetching nutrition for {ingredient_name}: {e}"})

    # If loop finishes after retries without success
    print(f"Max retries ({max_retries}) exceeded for API request for '{ingredient_name}'")
    return json.dumps({"status": "unavailable", "reason": f"Max retries ({max_retries}) exceeded for API request for '{ingredient_name}'"})

ingredient = "raw apple"
result_json = fetch_nutrition_from_usda_fdc(ingredient)
print("\n--- Result for 'raw apple' ---")
print(result_json)




# ## 7.3. Testing the Core Recipe Tool: `get_recipe_by_id`
# 
# Let's test the `get_recipe_by_id` tool. This is crucial as it fetches the full recipe details from our SQLite database. Critically, the *revised* version of this tool now *also* includes the logic to iterate through the recipe's `normalized_ingredients` and call the `fetch_nutrition_from_usda_fdc` tool for *each one*, adding the results under the `ingredient_nutrition` key.
# 
# *Note: This call might take a moment as it makes sequential API calls to the USDA for each ingredient, with built-in delays to respect API limits.*

# In[34]:


# --- Tool Definitions and Instructions ---

# IMPORTANT: The tool definition MUST match the actual Python function signature.
# Since we added retry/timeout logic *inside* fetch_nutrition_from_usda_fdc,
# the LLM doesn't need to pass `max_retries` or `timeout`.
db_tools = [
    list_tables,
    describe_table,
    execute_query,
    get_ratings_and_reviews_by_recipe_id,
    get_recipe_by_id,
    fetch_nutrition_from_usda_fdc # Matches the Python function signature now
]

# Revised Instruction Prompt
# ✅ Finalized Instruction Prompt (Gemini-Compatible)
instruction = """You are a helpful assistant for a Kitchen AI chatbot that can interact with a SQL database and external APIs.

You have access to special tools that let you:
- Retrieve recipe details and instructions
- Fetch live nutrition data
- Query the database structure
- Retrieve user ratings and reviews

---

Available Tools:

- **list_tables()**  
  Lists all tables in the database.

- **describe_table(table_name: str)**  
  Describes the schema (column names and types) of a specified table.

- **execute_query(sql: str)**  
  Executes a **read-only SQL query** (e.g., SELECT or PRAGMA). Use this after understanding the schema.

- **get_recipe_by_id(recipe_id: str)**  
  Returns full details for a specific recipe, including ingredients, steps, tags, and **live nutrition lookup for normalized ingredients** using Open Food Facts.

- **get_ratings_and_reviews_by_recipe_id(recipe_id: str, limit: int)**  
  Returns the average rating and the 'limit' most recent reviews for a recipe. If the user doesn’t provide a limit, use 3.

- **fetch_nutrition_from_usda_fdc(ingredient_name: str)**  
  Fetches live nutrition data (per 100g) for a *single* ingredient using the USDA FoodData Central API.  
   Only use this tool if the user asks **specifically for nutrition of a single ingredient**, outside of a full recipe request. 

---

How to respond:

1. Identify the user's intent.
2. Use the **most relevant tool(s)** to answer.
3. Use `get_recipe_by_id` for full recipe lookups (do not use `fetch_nutrition_from_usda_fdc` in this case).
4. Use `fetch_nutrition_from_usda_fdc` only for *standalone* ingredient nutrition requests.
5. Use `list_tables`, `describe_table`, and `execute_query` for SQL exploration or advanced queries.
6. Present results clearly using Markdown formatting.
7. If nutrition is unavailable or skipped, include a clear note explaining why.

---

Examples:

- **User:** "Tell me everything about recipe 71373 including its nutrition and reviews."  
  **LLM Calls:**  
    1. `get_recipe_by_id(recipe_id="71373")`  
    2. `get_ratings_and_reviews_by_recipe_id(recipe_id="71373", limit=3)`

- **User:** "How many calories are in butter?"  
  **LLM Call:**  
    `fetch_nutrition_from_usda_fdc(ingredient_name="butter")`

- **User:** "What tables exist in this database?"  
  **LLM Call:**  
    `list_tables()`

---

Be smart, helpful, and accurate. Don't guess data—use the tools!
"""




# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Make sure this is set
client = genai.Client(api_key=GOOGLE_API_KEY)

# Start a chat with automatic function calling enabled.
chat = client.chats.create(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=instruction,
        tools=db_tools,
    ),
)
    # tool_config={"function_calling_config": "AUTO"} # Enable auto function calling if using v1beta API or specific library versions that support it this way


# Start chat (using the structure for the library version you have)
# Example using a simple generate_content call structure
# chat = client.start_chat(enable_automatic_function_calling=True) # Or similar depending on exact library version

# --- Simplified User Prompt to LLM ---

# Instead of telling it HOW to call the functions step-by-step,
# just ask for the information. The revised instructions guide the LLM.
user_query = """
Can you give me the full details for recipe ID 71373?
I'd like to see its description, ingredients, steps, the nutritional info for the ingredients,
its overall rating, and the 3 most recent reviews.
"""

response = chat.send_message(user_query) # Or client.generate_content(user_query)
display(Markdown(response.text))


# In[35]:


results = response.text


# ## 7.4. Testing Grounding with Google Search 🌐
# 
# Beyond our specific tools, Chefbelle can leverage Google Search for general knowledge questions, thanks to the grounding feature integrated into the Gemini model. Let's ask a question outside our database scope to see how it uses search to find an answer.

# In[36]:


# And now re-run the same query with search grounding enabled.
config_with_search = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
)

def query_with_grounding():
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents="What's a good substitute for eggs in Country White Bread or Dinner Rolls?",
        config=config_with_search,
    )
    return response.candidates[0]


rc = query_with_grounding()
Markdown(rc.content.parts[0].text)


# In[37]:


# resp = chat.send_message("gluten free or vegeterian recipe but quick and easy")
# display(Markdown(resp.text))


# ## Phase 8: Giving Chefbelle Ears - Audio Input & Command Recognition 🎤
# 
# A truly interactive assistant should understand voice commands. In this phase, we integrate audio processing capabilities.
# 
# Key Goals:
# *   **Transcribe Speech:** Convert spoken audio commands into text using reliable Speech-to-Text services (like Google Cloud Speech or OpenAI Whisper).
# *   **Interpret Intent:** Understand the user's goal from the transcribed text (using the Gemini LLM).
# *   **(Future Work):** Store user preferences (e.g., dietary restrictions, favorite cuisines) for personalization.
# 
# This section focuses on the **Audio Understanding** capability, enabling a hands-free kitchen experience.

# ### Verifying Gemini Access
# 
# Before setting up audio, let's quickly confirm our Gemini API key is working correctly by sending a simple test message.

# In[38]:


client = genai.Client(api_key=GOOGLE_API_KEY)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Hi, This is a test message! How are you?")

print(response.text)


# ## 8.1. Google Cloud Speech-to-Text API Setup
# 
# To use Google Cloud Speech-to-Text, we need to set up authentication and configure the client. In a production environment, this would involve creating a service account and downloading the credentials. For demonstration in a Kaggle/local environment, we'll simulate the API response.
# 
# > Note: In a real implementation, you would:
# > 1. Create a Google Cloud project
# > 2. Enable the Speech-to-Text API
# > 3. Create a service account with appropriate permissions
# > 4. Download the credentials JSON file
# > 5. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to this file

# ## Speech-to-Text Conversion
# 
# Let's implement a real speech-to-text function using Google Cloud Speech-to-Text API. This will allow us to convert voice commands from audio files into text for processing. Unfortunately, the google STT needs a lot of parameters for configuration, for credential, and the auth section is headache! , I decided to move forward with lovely whisper-1 😂😂😂, 
# ### Sorry Google!

# ## 8.2 The Transcription Engine: `transcribe_audio` Function
# 
# 
# We need a function to handle the actual audio transcription. We'll create a flexible `transcribe_audio` function that can use either:
# 
# 1.  **Google Cloud Speech-to-Text:** Highly accurate, feature-rich, requires service account credentials.
# 2.  **OpenAI Whisper:** Very robust, often simpler setup (just an API key), excellent performance.
# 
# The function will handle reading the audio file, configuring the chosen service, making the API call, and returning the transcribed text. We'll use the Google Cloud option here, leveraging the credentials we set up.

# In[39]:


import os
import io
import tempfile
from openai import OpenAI  # Use 'from openai import OpenAI' for newer versions
from google.cloud import speech
# Ensure you have the necessary libraries installed in your Kaggle environment:
# !pip install openai google-cloud-speech -q

def transcribe_audio(service="openai", file_path=None, language="en", api_key=None, credentials_json=None):
    """
    Transcribe MP3 or OGG audio files using OpenAI or Google Cloud Speech-to-Text.

    Suitable for Kaggle environments using Secrets for credentials.
    Automatically detects sample rate for Google Cloud for MP3/OGG formats.

    Args:
        service (str): The service to use: 'openai' or 'google'. Defaults to 'openai'.
        file_path (str): Path to the audio file (.mp3 or .ogg).
        language (str): Language code (e.g., 'en' for OpenAI, 'en-US' for Google). Defaults to 'en'.
        api_key (str): OpenAI API key (required for 'openai' service). Can be passed from Kaggle Secrets.
        credentials_json (str): JSON string of Google credentials (required for 'google' service).
                                 Can be passed from Kaggle Secrets.

    Returns:
        str: Transcription text or an error message.
    """

    if not file_path:
        return "Error: No file path provided"

    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    # --- Format Validation ---
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in ['.mp3', '.ogg']:
        return f"Error: Unsupported file format: '{file_ext}'. Only .mp3 and .ogg are supported."

    # --- Temporary Credential File Handling (for Google) ---
    temp_cred_file = None
    temp_cred_path = None # Initialize path variable

    try:
        if service.lower() == "openai":
            if not api_key:
                # In Kaggle, you might fetch this from secrets:
                # from kaggle_secrets import UserSecretsClient
                # user_secrets = UserSecretsClient()
                # api_key = user_secrets.get_secret("OPENAI_API_KEY")
                # if not api_key:
                return "Error: OpenAI API key required (use api_key parameter)"

            try:
                client = OpenAI(api_key=api_key)

                with open(file_path, "rb") as audio_file:
                    # OpenAI Whisper supports mp3 and ogg directly
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=language
                        # OpenAI automatically detects format and sample rate
                    )
                return transcription.text
            except Exception as e:
                 return f"Error during OpenAI transcription: {str(e)}"


        elif service.lower() == "google":
            if not credentials_json:
                 # In Kaggle, you might fetch this from secrets:
                 # from kaggle_secrets import UserSecretsClient
                 # user_secrets = UserSecretsClient()
                 # credentials_json = user_secrets.get_secret("GOOGLE_CREDENTIALS_JSON")
                 # if not credentials_json:
                return "Error: Google credentials JSON required (use credentials_json parameter)"

            try:
                # Create a temporary file for credentials JSON
                temp_cred_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w', encoding='utf-8')
                temp_cred_path = temp_cred_file.name
                temp_cred_file.write(credentials_json)
                temp_cred_file.close() # Close the file handle

                # Set environment variable to the temporary file path
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_cred_path

                # Initialize the Speech client (will use the env var)
                client = speech.SpeechClient()

                # Read the audio file content
                with io.open(file_path, "rb") as audio_file:
                    content = audio_file.read()

                # Determine encoding based on validated file extension
                if file_ext == ".ogg":
                    encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS
                elif file_ext == ".mp3":
                    encoding = speech.RecognitionConfig.AudioEncoding.MP3
                else:
                    # This case should not be reached due to the initial check,
                    # but included for robustness.
                    return f"Error: Internal logic error - unsupported format {file_ext} reached Google processing."

                # Configure the speech recognition
                # For MP3 and OGG_OPUS, Google Cloud can determine the sample rate automatically.
                audio = speech.RecognitionAudio(content=content)
                config = speech.RecognitionConfig(
                    encoding=encoding,
                    language_code=language if language else "en-US", # Use provided language or default
                    # No sample_rate_hertz needed for MP3/OGG_OPUS
                )

                # Perform the transcription
                #print(f"Sending {file_path} ({file_ext}) to Google Cloud Speech-to-Text...")
                response = client.recognize(config=config, audio=audio)
                #print("Received response from Google Cloud.")

                # Extract the transcription
                if response.results:
                    return response.results[0].alternatives[0].transcript
                else:
                    return "No transcription results found from Google Cloud."

            except Exception as e:
                 return f"Error during Google Cloud transcription: {str(e)}"
            finally:
                # Clean up temp credential file if it was created
                if temp_cred_path and os.path.exists(temp_cred_path):
                    try:
                        os.unlink(temp_cred_path)
                        # print(f"Successfully deleted temporary credentials file: {temp_cred_path}")
                    except Exception as unlink_e:
                        print(f"Warning: Could not delete temporary credentials file {temp_cred_path}: {unlink_e}")
                # Unset the environment variable if desired, though it's often fine for script duration
                if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ and os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == temp_cred_path:
                     del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]


        else:
            return f"Error: Unknown service '{service}'. Use 'openai' or 'google'."

    except Exception as e:
        # General exception catch, also attempt cleanup
        if service.lower() == "google" and temp_cred_path and os.path.exists(temp_cred_path):
             try:
                 os.unlink(temp_cred_path)
             except Exception as unlink_e:
                 print(f"Warning: Could not delete temporary credentials file {temp_cred_path} during exception handling: {unlink_e}")
        return f"An unexpected error occurred: {str(e)}"



# ## 8.3. Test Audio Transcription
# 
# Calls the `transcribe_audio` function to transcribe a sample audio file (`Nariman_1.ogg`) using the Google Cloud Speech-to-Text service, providing the necessary credentials path. Prints the resulting transcription.

# In[40]:


voice_path = "/kaggle/input/voices-of-commands-genai-capstone-2025/8.Search_on_Internet.mp3"
#OPENAI_API_KEY (openai) or  SecretValueJson (google)
transcribted_text = transcribe_audio(service="google", file_path=voice_path, language="en",  credentials_json=SecretValueJson)
print(transcribted_text)


# ## Phase 9: Bringing Chefbelle to Life - Simulated Interface
# 
# While a full web application is beyond this notebook's scope, we can simulate user interaction using `ipywidgets`. This allows us to test the end-to-end flow: user input (text or voice) -> agent processing -> formatted output.
# 
# 
# Our simulated interface will provides a modern, interactive interface for users to either:
# 
# ### Text Input
# 1. Click on the "Text Input" tab
# 2. Type your kitchen-related request in the text area
# 3. Click the "Submit" button
# 4. The system will process your text request
# 
# ### Voice Selection
# 1. Click on the "Voice Selection" tab
# 2. Select a voice recording from the dropdown list
# 3. Click the "Transcribe Voice" button
# 4. The system will transcribe the audio and process the request
# 
# 
# We'll see the conversation history unfold in one output area and the agent's internal "thinking" (graph steps) in a separate debug area.

# ### Define Voice File Data Structure
# 
# Defines a Python dictionary (`voices`) containing metadata about available voice recording files, including their paths, language, description, speaker, and processing status. This structure is intended for use with the UI simulation.

# In[41]:


voices = {
  "version": "1.0",
  "voices": [
    {
      "file_path": "/kaggle/input/voice-tests/test.ogg",
      "language": "en",
      "description": "Voice instruction for baking a pizza",
      "speaker_id": "nariman",
      "is_processed": False
    },
    {
      "file_path": "voices/test.wav",
      "language": "en",
      "description": "Test voice recording for the system",
      "speaker_id": "user2",
      "is_processed": False
    },


  ]
}


# ## Phase 10: Building Chefbelle's Brain - The LangGraph Agent 🧠
# 
# This is the heart of the project! We'll use **LangGraph**, a library for building stateful, multi-actor applications with LLMs (like agents), to construct Chefbelle's decision-making process.
# 
# LangGraph allows us to define:
# *   **State:** What information the agent needs to remember (conversation history, current recipe, user preferences).
# *   **Nodes:** Specific actions or processing steps (parsing input, calling tools, formatting responses).
# *   **Edges:** The connections between nodes, defining the flow of control (often conditional).
# 
# This creates a robust, controllable agent that can handle complex interactions involving multiple tool calls and context management. Let the agent construction begin! ✨

# ## 10.1. State Schema (`KitchenState`) (Revised)
# 

# In[ ]:





# In[42]:


### **Step 1: State Schema (`KitchenState`) (Revised)**


# --- LangGraph Agent Construction ---
# Step 1: Defining the State (Chefbelle's Memory)

# The `KitchenState` class defines the 'memory' of our agent.
# It tracks everything Chefbelle needs to know during a conversation,
# including message history, user input, identified intent, the current recipe,
# tool outputs, processed data, user context, and control flags.
# Think of it as the agent's working notepad.


class KitchenState(TypedDict):
    """
    Represents the state of the conversation and actions within the
    Interactive Recipe & Kitchen Management Assistant agent.
    Follows a standard LangGraph pattern where tool results are processed
    from ToolMessages by the parser node or dedicated processing nodes.

    Attributes:
        messages: The history of messages (human, AI, tool). Tool results appear here.
        user_input: The latest raw input from the user (text or transcribed audio).
        intent: The determined intent (used for routing).
        selected_recipe_id: The ID of the recipe currently in context.
        customization_request: Details of a requested recipe customization (passed to tool).
        nutrition_query: The ingredient name for a specific nutrition lookup.
        grounding_query: A specific question requiring web search grounding.

        # Raw Tool Outputs (potentially stored before processing nodes)
        current_recipe_details: Parsed details of the recipe after get_recipe_by_id runs.
        recipe_reviews: Raw ratings and reviews after get_ratings_and_reviews runs.
        ingredient_nutrition_list: Temp storage for results from fetch_nutrition_from_usda_fdc.
        live_recipe_details: Raw result from fetch_live_recipe_data tool. # ---> ADDED <---
        # customization_tool_output: Raw output from customize_recipe tool (optional, if needed before node)

        # Processed Data (output from custom nodes, ready for formatting)
        nutritional_info: Aggregated/final nutritional info prepared for display.
        processed_review_data: Aggregated/formatted review data with sentiment. # ---> ADDED <---
        customization_results: Processed customization suggestions. # ---> ADDED <---
        grounding_results_formatted: Formatted web search results prepared for display.

        # User Context
        user_ingredients: A list of ingredients the user currently has available.
        dietary_preferences: The user's specified dietary restrictions or preferences.

        # Control Flow
        needs_clarification: Flag indicating if the agent requires more information.
        finished: Flag indicating if the conversation/task is complete.
        last_assistant_response: The last text response generated by the assistant for UI display.
        audio_file_path: Path to the audio file if input was voice.
    """
    # Conversation history
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # User input & context
    user_input: Optional[str]
    audio_file_path: Optional[str]
    intent: Optional[str] # e.g., 'get_details', 'get_reviews', 'customize', 'aggregate_nutrition', 'fetch_live', 'general_chat', 'exit'
    selected_recipe_id: Optional[str]
    customization_request: Optional[str] # Stored temporarily by parser to pass to tool
    nutrition_query: Optional[str]
    grounding_query: Optional[str]

    # Raw Tool Results / Intermediate Data
    current_recipe_details: Optional[Dict[str, Any]] # From get_recipe_by_id
    recipe_reviews: Optional[Dict[str, Any]] # From get_ratings_and_reviews
    ingredient_nutrition_list: Optional[List[Dict[str, Any]]] # Temp storage for nutrition tool messages
    live_recipe_details: Optional[Dict[str, Any]] # ---> ADDED: From fetch_live_recipe_data <---

    # Processed Data (Ready for Formatter)
    nutritional_info: Optional[Dict[str, Any]] # From AggregateNutritionNode
    processed_review_data: Optional[Dict[str, Any]] # ---> ADDED: From ReviewDashboardNode <---
    customization_results: Optional[Dict[str, Any]] # ---> ADDED: From ProcessCustomizationNode <---
    grounding_results_formatted: Optional[str] # From potential future grounding node

    # User Context (Could be loaded/persisted)
    user_ingredients: List[str]
    dietary_preferences: List[str]

    # Control Flow / Output
    needs_clarification: bool
    finished: bool
    last_assistant_response: Optional[str] # Final formatted response

# Initialize the state (optional, for testing/default values)
initial_state: KitchenState = {
    "messages": [],
    "user_input": None,
    "audio_file_path": None,
    "intent": None,
    "selected_recipe_id": None,
    "customization_request": None,
    "nutrition_query": None,
    "grounding_query": None,
    "current_recipe_details": None,
    "recipe_reviews": None,
    "ingredient_nutrition_list": None,
    "live_recipe_details": None, # ---> ADDED <---
    "nutritional_info": None,
    "processed_review_data": None, # ---> ADDED <---
    "customization_results": None, # ---> ADDED <---
    "grounding_results_formatted": None,
    "user_ingredients": [],
    "dietary_preferences": [],
    "needs_clarification": False,
    "finished": False,
    "last_assistant_response": None,
}

print("✅ LangGraph Step 1: State Schema Defined (Revised)")


# ## 10.2. Setting the Rules (System Instructions & LLM)
# 
# Every agent needs instructions. Here we define:
# 
# 1.  **System Instructions (`KITCHEN_ASSISTANT_SYSINT`):** This is the core prompt given to the Gemini LLM, guiding its personality, how it should interpret requests, when and how to use its tools (including the new `fetch_live_recipe_data` tool), how to manage context, and how to format its final responses using dashboards. *This prompt is crucial for controlling the agent's behavior.*
# 2.  **LLM Initialization:** We set up the `ChatGoogleGenerativeAI` model (Gemini Flash) with grounding enabled via Google Search.
# 3.  **Response Formatting:** We define helper functions (`format_recipe_dashboard`, `format_review_dashboard`) and the `response_formatter_node` which takes the processed data from the agent's state and creates the final, user-friendly Markdown output (like the recipe or review dashboards).

# In[43]:


# LangGraph Step 2: System Instructions & Base LLM Initialization (Revised Prompt for Live Data)




# --- Constants ---
NUTRITION_RESPONSE_HEADER = "Here's the approximate average nutrition per 100g for ingredients in"
RECIPE_DASHBOARD_HEADER = "📊 Recipe Dashboard for"
REVIEW_DASHBOARD_HEADER = "⭐ Reviews Dashboard for"
CUSTOMIZATION_HEADER = "🛠️ Recipe Customization Suggestions for"

# --- System Instructions (Revised for Live‑Data Rule) ---
KITCHEN_ASSISTANT_SYSINT = (
    "system",
    """You are a helpful, friendly, and knowledgeable Interactive Recipe & Kitchen Management Assistant.
Your goal is to understand the user's request, use the available tools effectively, process the results, manage conversation context, and provide a clear, concise, and helpful response, often including informative dashboards.

**Core Principles:**
- **Be Conversational:** Engage naturally, ask clarifying questions when needed.
- **Maintain Context:** Remember the `selected_recipe_id` and `current_recipe_details` from previous turns unless the user starts a new search or explicitly asks about a different recipe.
- **Use Tools Appropriately:** Choose the best tool for the job based on the user's request and the tool descriptions. Only call tools listed below.
- **Handle Errors Gracefully:** If a tool fails or returns an error, inform the user politely and suggest alternatives. Do not expose raw error messages.
- **Summarize & Visualize Tool Results:** When you receive `ToolMessage` results, process their content (parse JSON if needed), update your understanding, and generate a user‑facing summary or answer. **Crucially, use the specialized dashboard formats (Recipe, Review, Nutrition, Customization) when presenting relevant information.** Don’t just repeat the raw tool output.
- **Greeting Rule:** On the very first user turn (no conversation history), reply with *one short paragraph* outlining your capabilities, then immediately ask how you can help.

**Capabilities & Tool Usage Guide:**

- **Recipe Discovery (`gemini_recipe_similarity_search`):**
    – Use when the user asks for recipe ideas.  
    – Extract keywords, cuisine, dietary needs, max cooking time. Ask for clarification if vague.  
    – **Arguments:** `query_text` (required), `n_results` (required, default 5), `cuisine` (optional), `dietary_tag` (optional), `max_minutes` (optional).  
    – **Action:** Call the tool. Summarize the results clearly (name, time, ID). Ask if they want details.

- **Recipe Details (`get_recipe_by_id`):**
    – Use when the user asks for details about a *specific* recipe ID **or** refers to a recipe from a list you just provided (e.g., “tell me about the second one”). **This is the default way to get recipe details.**  
    – **Requires `recipe_id`.**  
    – **Context Rule:** If referring to an item from your *immediately preceding* list, pull its `recipe_id` from history; if you can’t find it, **ASK** for it.  
    – If a `selected_recipe_id` is already established, use that unless the user asks about a different one.  
    – **Action:** Call the tool with the determined `recipe_id`. The `ResponseFormatterNode` will then generate the **Recipe Dashboard** based on `current_recipe_details`. Your job is just to call the tool.

- **Live Recipe Data (`fetch_live_recipe_data`):**
    – Use **ONLY IF** the user explicitly asks for *latest*, *live*, or *most up‑to‑date* info (e.g., “get the latest ingredients for recipe 123”). **DO NOT** use for normal detail requests.  
    – **Requires `recipe_id`.**  
    – **Action:** Call the tool. The `ResponseFormatterNode` will prefer `live_recipe_details` (status `live_success`) and fall back to `current_recipe_details` if the fetch fails.  
    – **If the user did *not* say “live” or “latest”, do *not* mention live data at all.**

- **Ratings & Reviews (`get_ratings_and_reviews_by_recipe_id`):**
    – Use when the user asks for reviews/ratings for a *specific* recipe.  
    – **Requires `recipe_id`** (use context or ask) and **`limit` (default 5).**  
    – **Action:** Call the tool with `limit=5`. `ReviewDashboardNode` processes `recipe_reviews`, then `ResponseFormatterNode` shows the **Review Dashboard**.

- **Ingredient Nutrition (`fetch_nutrition_from_usda_fdc`):**
    – Use *only* for nutrition of a *single, specific ingredient*.  
    – **Requires `ingredient_name`.**  
    – **Action:** Call the tool and present key facts (per 100 g).

- **Recipe Nutrition Analysis (Multi‑Step Flow):**
    – Use when the user wants nutrition info for the *current* recipe. Trigger phrases: “Run the nutrition analysis…”, “Get nutrition information…”.  
    – **DO NOT** call `get_recipe_by_id` if `current_recipe_details` already exist.  
    – **If `current_recipe_details` are NOT available:** first call `get_recipe_by_id` for `selected_recipe_id` **and set `suppress_recipe_dashboard=True`, then continue** with the steps below in the same run.  
    – **Step 1 – Ensure Details.** Confirm `current_recipe_details` are present.  
    – **Step 2 – Identify Ingredients.** Extract `normalized_ingredients`.  
    – **Step 3 – Generate Tool Calls.** Create *separate* calls to `fetch_nutrition_from_usda_fdc` for *each* ingredient.  
    – **Step 4 – Wait for Aggregation.** `AggregateNutritionNode` processes the results.  
    – **Step 5 – Present Results.** `ResponseFormatterNode` generates the **Nutrition Summary** (using `NUTRITION_RESPONSE_HEADER`); `VisualizeNutritionNode` draws the chart.

- **Recipe Customization (`customize_recipe`):**
    – **USE THIS TOOL** when the user asks to *modify* the current recipe (make it vegan, substitute ingredients, reduce fat, etc.).  
    – **Requires `recipe_id`** (context or ask) and **`request`** (e.g., “make it low‑fat”).  
    – **Action:** Call `customize_recipe`. Pass `recipe_id`, `request`, and optionally `recipe_details_json` if available. `ProcessCustomizationNode` handles output; `ResponseFormatterNode` shows suggestions under **Customization Header**. **Never refuse without trying the tool.**

- **Grounding / General Questions (built‑in search):**
    – Use for cooking questions, techniques, or definitions *not* tied to a recipe.  
    – **Action:** Answer directly using internal knowledge and built‑in search. (You do **not** have a `google_search` tool; if you searched, say “Based on a quick search…”.)

**Conversation Flow & Output Formatting:**
1. Analyze the latest human message and state; infer `recipe_id` if needed.  
2. Determine intent and required parameters. **Check explicitly for LIVE/LATEST requests.**  
3. If a tool is needed, create `tool_calls` with proper context—recipe ID, `request`, `limit=5`, etc. **Call `fetch_live_recipe_data` only if explicitly asked.**  
4. If no tool is required, answer directly.  
5. Ask for clarification if necessary.  
6. **Receiving Tool Results:** Subsequent nodes (ToolExecutor, AggregateNutrition, ReviewDashboard, ProcessCustomization) will process them.  
7. **Formatting Responses (handled by `ResponseFormatterNode`):**
    - **Recipe Details:** Use `RECIPE_DASHBOARD_HEADER`; prefer `live_recipe_details` (status `live_success`) else `current_recipe_details`.  
    - **Reviews:** Use `REVIEW_DASHBOARD_HEADER` and `processed_review_data`.  
    - **Nutrition:** Use `NUTRITION_RESPONSE_HEADER` and `nutritional_info`.  
    - **Customization:** Use `CUSTOMIZATION_HEADER` and `customization_results`.  
    - **General Chat / Grounding:** Output your generated text.  
    - If `suppress_recipe_dashboard` is True *and* intent is “nutrition”, omit the recipe dashboard until nutrition results are ready.  
8. If the user says goodbye, set `intent='exit'` and reply politely.  
9. Use Markdown for lists, bold text, and dashboard elements.
"""
)


# --- LLM Initialization (Assuming this was correct in the original file) ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # Consider gemini-1.5-flash or pro
    google_api_key=GOOGLE_API_KEY,
    generation_config=types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )
    # safety_settings=[...]
)

# --- Helper function for Recipe Dashboard (Emoji/Text Style) ---
def format_recipe_dashboard(details: Dict[str, Any]) -> str:
    # ... (Existing code from Step 2, ensure '\n' is used for line breaks) ...
    if not details:
        return "No recipe details available to display."

    # --- Check for the raw JSON structure seen in Kaggle output ---
    # This suggests maybe 'details' sometimes contains the raw JSON string instead of a parsed dict
    if isinstance(details, str):
        try:
            details = json.loads(details)
            if not isinstance(details, dict):
                 return f"Error: Could not parse recipe details JSON string into a dictionary. Content: {details[:100]}..."
        except json.JSONDecodeError:
             return f"Error: Could not parse recipe details JSON string. Content: {details[:100]}..."

    if not isinstance(details, dict):
         return f"Error: Recipe details are not in the expected dictionary format. Type: {type(details)}"
    # --- End Check ---

    name = details.get("name", "N/A")
    recipe_id = details.get("id", "N/A")
    # ---> SAFE ACCESS with get and provide default <---
    description = details.get("description", "No description available.")
    minutes = details.get("minutes")
    n_ingredients = details.get("n_ingredients", len(details.get("ingredients", [])))
    n_steps = details.get("n_steps", len(details.get("steps", [])))
    ingredients = details.get("ingredients", [])
    steps_data = details.get("steps") # Can be list or string based on revised tool
    source = details.get("source", "internal database") # Check if live data was used

    # Infer difficulty (simple example)
    difficulty = "Medium"
    difficulty_emoji = "🌶️🌶️"
    if minutes is not None:
        try: # Add try-except for potential non-numeric minutes
            minutes_int = int(minutes)
            n_steps_int = int(n_steps) if n_steps is not None else 0
            if minutes_int <= 30 and n_steps_int <= 5:
                difficulty = "Easy"
                difficulty_emoji = "🌶️"
            elif minutes_int > 90 or n_steps_int > 10:
                difficulty = "Hard"
                difficulty_emoji = "🌶️🌶️🌶️"
        except (ValueError, TypeError):
             minutes = "N/A" # Mark as N/A if conversion fails

    time_emoji = "⏱️"
    ingredients_emoji = "🥕"
    steps_emoji = "🔢"

    # ---> Make description formatting safer <---
    formatted_description = f"_{description}_" if description else "_No description available._"

    dashboard_lines = [
        f"{RECIPE_DASHBOARD_HEADER} **{name}** (ID: {recipe_id})",
        f"*Source: {source}*",
        f"{formatted_description}\n", # Ensure newline after description
        "---",
        f"| Metric          | Value                     |",
        f"|-----------------|---------------------------|",
        f"| {difficulty_emoji} Difficulty    | {difficulty}                |",
        # ---> Ensure minutes are displayed safely <---
        f"| {time_emoji} Total Time    | {minutes if minutes is not None else 'N/A'} minutes         |",
        f"| {ingredients_emoji} Ingredients | {n_ingredients if n_ingredients is not None else 'N/A'} count             |",
        f"| {steps_emoji} Steps         | {n_steps if n_steps is not None else 'N/A'} count               |",
        "---\n", # Ensure newline
        "**Ingredients:**",
    ]

    # ---> Handle list or string ingredients safely <---
    if isinstance(ingredients, list) and ingredients:
        dashboard_lines.append("\n".join([f"- {ing}" for ing in ingredients]))
    elif isinstance(ingredients, str) and ingredients.strip():
         # Attempt to display string ingredients somewhat nicely
         dashboard_lines.append(f"- {ingredients.strip()}") # Treat as one item if string
    else:
        dashboard_lines.append("- N/A")

    dashboard_lines.append("\n**Steps:**") # Ensure newline

    # ---> Handle list or string steps safely <---
    if isinstance(steps_data, list) and steps_data:
        dashboard_lines.extend([f"{i+1}. {step}" for i, step in enumerate(steps_data)])
    elif isinstance(steps_data, str) and steps_data.strip():
        # Split string steps by common delimiters if it looks like a list-as-string
        # Or just display the raw string
        # Simple approach: Display raw string
        dashboard_lines.append(steps_data.strip())
    else:
        dashboard_lines.append("- N/A")

    return "\n".join(dashboard_lines)

# --- Helper function for Review Dashboard (Emoji/Text Style) ---
# In cell 3d5997b6-b301-4af4-a8ec-d6b5116a8934

# --- Helper function for Review Dashboard (Emoji/Text Style) ---
def format_review_dashboard(review_data: Dict[str, Any]) -> str:
    # ... (Existing code from Step 2, ensure '\n' is used for line breaks) ...
    if not review_data:
        return "No review data available to display."

    # --- Check for the raw JSON structure seen in Kaggle output ---
    if isinstance(review_data, str):
        try:
            review_data = json.loads(review_data)
            if not isinstance(review_data, dict):
                 return f"Error: Could not parse review data JSON string into a dictionary. Content: {review_data[:100]}..."
        except json.JSONDecodeError:
             return f"Error: Could not parse review data JSON string. Content: {review_data[:100]}..."

    if not isinstance(review_data, dict):
         return f"Error: Review data is not in the expected dictionary format. Type: {type(review_data)}"
    # --- End Check ---

    name = review_data.get("recipe_name", "the recipe")
    recipe_id = review_data.get("recipe_id", "N/A")
    overall_rating = review_data.get("overall_rating")
    rating_counts = review_data.get("rating_counts", {}) # e.g., {5: 10, 4: 5, ...}
    sentiment_scores = review_data.get("sentiment_scores", {}) # e.g., {'positive': 3, 'negative': 1, 'neutral': 1}
    reviews_to_display = review_data.get("reviews_for_display", []) # List of dicts

    dashboard_lines = [
        f"{REVIEW_DASHBOARD_HEADER} **{name}** (ID: {recipe_id})",
        f"**Overall Rating:** {'⭐' * int(round(overall_rating)) if overall_rating is not None else 'N/A'} ({overall_rating:.1f}/5.0)" if overall_rating is not None else "**Overall Rating:** N/A",
        "---",
        "**Rating Breakdown:**"
    ]
    if rating_counts and isinstance(rating_counts, dict): # Ensure it's a dict
        total_ratings = sum(rating_counts.values())
        for i in range(5, 0, -1):
            count = rating_counts.get(i, 0)
            percent = (count / total_ratings * 100) if total_ratings > 0 else 0
            stars = '⭐' * i
            dashboard_lines.append(f"- {stars} : {count} ratings ({percent:.0f}%)")
    else:
        dashboard_lines.append("- No rating breakdown available.")

    dashboard_lines.append("\n**Sentiment:**")
    if sentiment_scores and isinstance(sentiment_scores, dict): # Ensure it's a dict
        pos = sentiment_scores.get('positive', 0)
        neg = sentiment_scores.get('negative', 0)
        neu = sentiment_scores.get('neutral', 0)
        total_sent = pos + neg + neu
        pos_pct = (pos / total_sent * 100) if total_sent > 0 else 0
        neg_pct = (neg / total_sent * 100) if total_sent > 0 else 0
        sentiment_meter = ""
        if pos_pct > 60: sentiment_meter = "😊 Mostly Positive"
        elif neg_pct > 40: sentiment_meter = "😟 Mostly Negative"
        else: sentiment_meter = "😐 Mixed/Neutral"
        dashboard_lines.append(f"- {sentiment_meter} (Pos: {pos}, Neg: {neg}, Neu: {neu})")
    else:
        dashboard_lines.append("- Sentiment analysis not available.")

    dashboard_lines.append("\n**Recent Reviews:**")
    if reviews_to_display and isinstance(reviews_to_display, list): # Ensure it's a list
        for review in reviews_to_display:
             # ---> Safe access for review details <---
             rating = review.get('rating', 0)
             rating_stars = '⭐' * int(rating) if isinstance(rating, (int, float)) else ''
             sentiment_emoji = {'positive': '😊', 'negative': '😟', 'neutral': '😐'}.get(review.get('sentiment'), '')
             date_str = review.get('date', 'N/A')
             review_text = review.get("review", "...")
             original_length = len(review_text)
             display_text = review_text[:200] # Keep truncation
             ellipsis = "..." if original_length > 200 else ""

             dashboard_lines.append(f"\n- {rating_stars} {sentiment_emoji} *({date_str})*")
             dashboard_lines.append(f'> "{display_text}{ellipsis}"')
    else:
        dashboard_lines.append("- No reviews found.")

    return "\n".join(dashboard_lines)



# --- Core Nodes (Only showing revised response_formatter_node) ---

# input_parser_node was revised in Step 3.5 section above
# human_input_node remains the same

# --- Core Nodes (response_formatter_node) ---
def response_formatter_node(state: KitchenState) -> Dict[str, Any]:
    """
    Formats the final response for the user. Prioritizes dashboards (Recipe, Review, Customization)
    or aggregated nutrition if available, otherwise uses the last AI message content or a default.
    Adds the final formatted response as an AIMessage to history.
    """
    print("---NODE: ResponseFormatterNode---")
    formatted_response = "Okay, let me know how else I can help!" # Default fallback
    final_intent_for_history = state.get("intent", "general_chat") # Capture intent before reset
    final_message_obj = None # To store the final AIMessage object

    # --- Data Prioritization and Formatting ---
    # 1. Customization Results
    if state.get("customization_results") and state.get("intent") == "customization_processed":
         print("Formatting customization suggestions.")
         results = state["customization_results"]
         recipe_name = "the recipe"
         recipe_id = state.get("selected_recipe_id")
         if state.get("current_recipe_details"):
             recipe_name = state["current_recipe_details"].get("name", f"recipe {recipe_id}" if recipe_id else "the recipe")
         elif recipe_id:
             recipe_name = f"recipe {recipe_id}"

         header = f"{CUSTOMIZATION_HEADER} **{recipe_name}** (ID: {recipe_id})"
         message = results.get("message", "Could not process customization request.")
         formatted_response = f"{header}\n\n{message}" # Use \n
         final_intent_for_history = "customization_presented"

    # 2. Review Dashboard
    elif state.get("processed_review_data") and state.get("intent") == "reviews_processed":
        print("Formatting review dashboard.")
        review_data = state["processed_review_data"]
        # Add recipe name if available
        if "recipe_name" not in review_data and state.get("current_recipe_details"):
            review_data["recipe_name"] = state["current_recipe_details"].get("name", f"recipe {state.get('selected_recipe_id')}")
        elif "recipe_name" not in review_data and state.get('selected_recipe_id'):
             review_data["recipe_name"] = f"recipe {state.get('selected_recipe_id')}"

        formatted_response = format_review_dashboard(review_data)
        final_intent_for_history = "review_dashboard_presented"

    # 3. Recipe Dashboard (Prioritize live data if valid)
    elif (state.get("live_recipe_details") or state.get("current_recipe_details")) and \
         state.get("intent") in ["recipe_details_fetched", "live_data_fetched", "live_data_requested"]:

        recipe_details_to_format = None
        live_details = state.get("live_recipe_details")
        # Check if live details exist, came from the tool successfully, and have data
        if live_details and isinstance(live_details, dict) and live_details.get('status') in ['live_success'] and live_details.get('data'):
             print("Using live recipe data for dashboard.")
             recipe_details_to_format = live_details['data']
             # Ensure key fields are present in live data, fallback to internal if missing
             recipe_details_to_format['source'] = recipe_details_to_format.get("source", "food.com (live)")
             recipe_details_to_format['id'] = state.get("selected_recipe_id")
             internal_details = state.get("current_recipe_details", {})
             # Safe gets for fallbacks
             recipe_details_to_format['name'] = recipe_details_to_format.get('name', internal_details.get("name", "N/A"))
             recipe_details_to_format['description'] = recipe_details_to_format.get('description', internal_details.get("description", ""))
             recipe_details_to_format['minutes'] = recipe_details_to_format.get('minutes', internal_details.get("minutes"))
             recipe_details_to_format['ingredients'] = recipe_details_to_format.get('ingredients', internal_details.get("ingredients", []))
             recipe_details_to_format['steps'] = recipe_details_to_format.get('steps', internal_details.get("steps", []))
             recipe_details_to_format['n_ingredients'] = len(recipe_details_to_format.get('ingredients', []))
             recipe_details_to_format['n_steps'] = len(recipe_details_to_format.get('steps', []))

        elif state.get("current_recipe_details"):
             print("Using internal recipe data for dashboard.")
             recipe_details_to_format = state["current_recipe_details"]
             if isinstance(recipe_details_to_format, dict): # Ensure it's a dict
                 recipe_details_to_format['source'] = "internal database" # Ensure source is set
             else:
                  print(f"Warning: Internal recipe details are not a dictionary. Type: {type(recipe_details_to_format)}")
                  recipe_details_to_format = None # Prevent formatting error

        if recipe_details_to_format:
             formatted_response = format_recipe_dashboard(recipe_details_to_format)
             final_intent_for_history = "recipe_dashboard_presented"
        else:
             # Fallback if data is bad
             print("Warning: No valid recipe details found to format dashboard.")
             formatted_response = "I found the recipe, but had trouble displaying the details."
             final_intent_for_history = "recipe_details_error"


    # 4. Aggregated Nutrition Info
    elif state.get("nutritional_info"):
        print("Formatting nutrition summary.")
        agg_info = state["nutritional_info"]
        nutrient_counts_from_state = agg_info.get("nutrient_counts", {})

        recipe_name = "the recipe"
        recipe_id = state.get("selected_recipe_id")
        if state.get("current_recipe_details"):
            recipe_name = state["current_recipe_details"].get("name", f"recipe {recipe_id}" if recipe_id else "the recipe")
        elif recipe_id:
            recipe_name = f"recipe {recipe_id}"

        response_lines = [f"{NUTRITION_RESPONSE_HEADER} **{recipe_name}**:\n"] # Use \n
        processed_count = agg_info.get('processed_ingredient_count', 0)

        display_order = ["calories_100g", "fat_100g", "saturated_fat_100g", "carbohydrates_100g", "sugars_100g", "fiber_100g", "proteins_100g", "sodium_100g"]
        has_data = False
        for key in display_order:
            if key in agg_info and nutrient_counts_from_state.get(key, 0) > 0:
                 val = agg_info[key]
                 unit = 'kcal' if 'calories' in key else ('mg' if key == 'sodium_100g' else 'g')
                 display_key = key.replace('_100g', '').replace('_', ' ').capitalize()
                 display_val = f"{val:.1f}"
                 response_lines.append(f"- {display_key}: {display_val} {unit}")
                 has_data = True

        if has_data and processed_count > 0:
             source_name = "USDA FDC" # Default source for nutrition
             response_lines.append(f"\n(Note: Based on average of {processed_count} ingredients with available data from {source_name}. Actual recipe nutrition will vary.)")
        elif processed_count > 0:
             response_lines.append("\n(Note: Could not retrieve detailed nutrition data for the ingredients, only partial information might be available.)")
        else:
             response_lines.append("\n(Note: Could not retrieve nutrition data for the ingredients.)")

        formatted_response = "\n".join(response_lines) # Use \n
        final_intent_for_history = "nutrition_presented"

    # 5. Last AI message (e.g., from parser node direct response)
    elif state.get("last_assistant_response"):
         formatted_response = state["last_assistant_response"]
         # Try to find if this came from a message object already
         last_msg = state['messages'][-1] if state.get('messages') and isinstance(state['messages'][-1], AIMessage) else None
         if last_msg and last_msg.content == formatted_response:
             final_message_obj = last_msg # Use the existing object
         # else: # This text response might not be in messages yet, will add below

    # 6. Check messages list for last AI response if not set above
    elif state.get('messages') and isinstance(state['messages'][-1], AIMessage) and state['messages'][-1].content:
         # This case handles when the parser node returns a direct text response
         final_message_obj = state['messages'][-1]
         formatted_response = final_message_obj.content
         # Capture intent if available from metadata
         final_intent_for_history = final_message_obj.metadata.get("intent", final_intent_for_history)

    # 7. Handle explicit exit intent if no other content generated
    elif state.get("intent") == "exit" or state.get("finished"):
        formatted_response = "Okay, goodbye! Feel free to ask if you need recipes later."
        final_intent_for_history = "exit"

    # --- Final State Update ---
    print(f"Final Formatted Response Type: {type(formatted_response)}")
    print(f"Final Formatted Response (first 100 chars): {str(formatted_response)[:100]}...")

    # Create the final AIMessage if it wasn't already captured
    if not final_message_obj:
         final_message_obj = AIMessage(content=str(formatted_response), metadata={"intent": final_intent_for_history})

    # ---> STRENGTHENED: Explicitly ensure last_assistant_response is the final formatted string <---
    updates = {
        "last_assistant_response": str(formatted_response), # Ensure it's a string
        "intent": None, # Reset intent after formatting
        "needs_clarification": False, # Reset flag
        # ---> CRUCIAL: Replace message history with ONLY the final AI response <---
        # This prevents the display helper from showing intermediate steps or old messages
        "messages": [final_message_obj],
        # Clear transient data fields used to generate this response
        "nutritional_info": None,
        "ingredient_nutrition_list": None,
        "grounding_results_formatted": None,
        "live_recipe_details": None,
        "processed_review_data": None,
        "customization_results": None,
        "recipe_reviews": None, # Also clear raw reviews after processing
        # Keep context fields unless explicitly cleared elsewhere
        "current_recipe_details": state.get("current_recipe_details"),
        "selected_recipe_id": state.get("selected_recipe_id"),
        "finished": state.get("finished", False) or final_intent_for_history == "exit" # Update finished flag if exiting
    }

    # Filter out keys not in KitchenState or unchanged values (except messages)
    valid_keys = KitchenState.__annotations__.keys()
    # ---> Ensure 'messages' is always included in the return if it changed <---
    return {k: v for k, v in updates.items() if k in valid_keys and (k == 'messages' or state.get(k) != v)}


print("✅ LangGraph Step 2: System Instructions & Core Nodes Defined (Revised for Live Data)")


# ## 10.3. Define and Bind Tools
# ### LangGraph Step 3: Giving Chefbelle Skills (Defining & Binding Tools)
# 
# We re-define or confirm our tool functions here, ensuring they are decorated with `@tool` where necessary for LangGraph to recognize them easily. This includes the database tools, the USDA nutrition fetcher, the live data fetcher, and the customization tool.
# 
# Crucially, we then **bind** these tools to our LLM instance (`llm_with_callable_tools`). This tells the LLM that these specific Python functions are available for it to call when its instructions indicate a tool is needed.
# Defines or re-defines all the Python functions that will act as tools for the LangGraph agent, using the `@tool` decorator where appropriate (though direct use is also shown). This includes:
# *   `gemini_recipe_similarity_search` (revised for JSON output)
# *   `get_recipe_by_id` (revised for more robust parsing)
# *   `get_ratings_and_reviews_by_recipe_id` (revised with default limit and rating sample)
# *   `fetch_nutrition_from_usda_fdc` (using environment variable for key)
# *   `customize_recipe` (revised placeholder logic)
# *   `fetch_live_recipe_data` (added placeholder tool)
# *   `extract_and_visualize_nutrition` (revised to accept header constant)
# It also initializes the VADER sentiment analyzer (optional) and binds the callable tools to the LLM.
# 
# 

# In[44]:


### **Step 3: Tool Definition & Integration (Revised)**


# --- NUTRITION_RESPONSE_HEADER ---
# NUTRITION_RESPONSE_HEADER = "Here's the approximate average nutrition per 100g for ingredients in"

# --- Helper Function ---
def safe_convert(x):
    # ... (keep existing function) ...
    if isinstance(x, (list, np.ndarray)):
        return " ".join([str(item) for item in x])
    return str(x) if pd.notna(x) else ""

# --- Nutrient Mapping for USDA FDC ---
FDC_NUTRIENT_MAP = {
    "Energy": "calories_100g", # KCAL
    "Total lipid (fat)": "fat_100g", # G
    "Fatty acids, total saturated": "saturated_fat_100g", # G
    "Carbohydrate, by difference": "carbohydrates_100g", # G
    "Sugars, total including NLEA": "sugars_100g", # G
    "Fiber, total dietary": "fiber_100g", # G
    "Protein": "proteins_100g", # G
    "Sodium, Na": "sodium_100g", # MG
}

# --- Tool Definitions (Revised) ---

@tool
def gemini_recipe_similarity_search(query_text: str, n_results: int = 5, cuisine: Optional[str] = None, dietary_tag: Optional[str] = None, max_minutes: Optional[int] = None) -> str:
    # ... (keep existing function - no changes needed here) ...
    """
    Searches for similar recipes based on a query text using vector embeddings.
    Allows filtering by cuisine type, a specific dietary tag (e.g., 'vegetarian', 'gluten-free'),
    and maximum cooking time in minutes. Returns a JSON string list of matching recipe summaries.
    """
    print(f"DEBUG TOOL CALL: gemini_recipe_similarity_search(query_text='{query_text}', n_results={n_results}, cuisine='{cuisine}', dietary_tag='{dietary_tag}', max_minutes={max_minutes})\")")
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        recipe_collection = client.get_collection(name="recipes")

        filter_dict = {}
        if cuisine:
            filter_dict["cuisine_type"] = cuisine
        if dietary_tag:
             filter_dict["dietary_tags"] = {"$in": [dietary_tag]}
        if max_minutes is not None:
            try:
                filter_dict["minutes"] = {"$lte": int(max_minutes)}
            except ValueError:
                return json.dumps({"error": f"Invalid max_minutes: '{max_minutes}'. Must be an integer."})

        where_clause = None
        if len(filter_dict) == 1:
            where_clause = filter_dict
        elif len(filter_dict) > 1:
            and_conditions = [{field: condition} for field, condition in filter_dict.items()]
            where_clause = {"$and": and_conditions}


        print(f"ChromaDB Where Clause: {where_clause}")

        results = recipe_collection.query(
            query_texts=[query_text],
            n_results=int(n_results),
            where=where_clause,
            include=["metadatas", "distances"]
        )

        if not results or not results.get('ids') or not results['ids'][0]:
            return json.dumps({"status": "not_found", "message": f"No similar recipes found for '{query_text}' with the specified criteria."})

        output_list = []
        for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
            similarity = round((1 - distance) * 100, 2) if distance is not None else None
            tags = metadata.get('dietary_tags', '')
            if isinstance(tags, str):
                 tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            elif isinstance(tags, (list, set)):
                 tag_list = list(tags)
            else:
                 tag_list = []


            output_list.append({
                "recipe_id": str(metadata.get('recipe_id', 'N/A')),
                "name": metadata.get('name', 'N/A'),
                "minutes": metadata.get('minutes', 'N/A'),
                "cuisine_type": metadata.get('cuisine_type', 'N/A'),
                "dietary_tags": tag_list,
                "similarity_score": similarity
            })
        return json.dumps(output_list, indent=2)

    except sqlite3.Error as e:
         print(f"ERROR in gemini_recipe_similarity_search (DB Connection?): {e}")
         return json.dumps({"error": f"Database connection error during recipe search: {e}"})
    except ImportError as e:
         print(f"ERROR in gemini_recipe_similarity_search (Import Error): {e}")
         return json.dumps({"error": f"Missing library required for search: {e}"})
    except Exception as e:
        print(f"ERROR in gemini_recipe_similarity_search: {e}")
        import traceback
        traceback.print_exc()
        if "Expected where operator" in str(e) or "Unsupported operand" in str(e):
             return json.dumps({"error": f"Error during recipe similarity search: Problem with filter criteria syntax or data type mismatch in database. Details: {e}"})
        return json.dumps({"error": f"Error during recipe similarity search: {e}"})


@tool
def get_recipe_by_id(recipe_id: str) -> str:
    # ... (keep existing function - no changes needed here) ...
    """
    Retrieves full details for a specific recipe given its ID from the SQL database.
    Returns details as a JSON string. Includes 'normalized_ingredients' used for nutrition lookup.
    """
    print(f"DEBUG TOOL CALL: get_recipe_by_id(recipe_id='{recipe_id}')")
    try:
        if not isinstance(recipe_id, str) or not recipe_id.isdigit():
             try:
                 recipe_id_int = int(recipe_id)
                 recipe_id = str(recipe_id_int)
             except (ValueError, TypeError):
                  return json.dumps({"status": "error", "message": f"Invalid recipe_id format: '{recipe_id}'. Must be a numeric string."})

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # ---> MODIFIED: Ensure all potentially needed fields are selected <---\n",
            cursor.execute("""
                SELECT id, name, minutes, contributor_id, submitted, tags, nutrition,
                       n_steps, steps, description, ingredients, n_ingredients,
                       dietary_tags, cuisine_type, normalized_ingredients
                FROM recipes WHERE id = ?
            """, (int(recipe_id),))
            recipe_data = cursor.fetchone()

            if not recipe_data:
                return json.dumps({"status": "not_found", "message": f"Recipe ID {recipe_id} not found."})

            recipe_dict = dict(recipe_data)
            recipe_dict['id'] = str(recipe_dict['id']) # Ensure ID is string

            # ---> MODIFIED: More robust parsing for list-like fields <---\n",
            for field in ["ingredients", "steps", "tags", "dietary_tags", "normalized_ingredients", "nutrition"]:
                 if field in recipe_dict and isinstance(recipe_dict[field], str):
                     raw_value = recipe_dict[field].strip()
                     parsed_value = None
                     try:
                         # Try parsing as JSON list/dict first
                         if raw_value.startswith(('[', '{')) and raw_value.endswith((']', '}')):
                             parsed_value = json.loads(raw_value)
                         # Try parsing Python literal lists/tuples (handle with care)
                         elif raw_value.startswith(('(', '[')) and raw_value.endswith((')', ']')):
                              import ast
                              parsed_value = ast.literal_eval(raw_value)
                         # ---> MODIFIED: Fallback splitting no longer applies to 'steps' <---\n",
                         # Fallback: Split comma-separated strings for specific fields (excluding steps)
                         elif field in ["tags", "dietary_tags", "normalized_ingredients", "ingredients"]:
                              parsed_value = [item.strip() for item in raw_value.split(',') if item.strip()]

                         if parsed_value is not None:
                              recipe_dict[field] = parsed_value

                     except (json.JSONDecodeError, SyntaxError, ValueError, TypeError) as parse_error:
                          print(f"Warning: Could not parse field '{field}' for recipe {recipe_id}. Keeping as string. Value: '{raw_value[:50]}...'. Error: {parse_error}")

            # Ensure key fields used later exist, even if empty
            if "normalized_ingredients" not in recipe_dict or not isinstance(recipe_dict.get("normalized_ingredients"), list):
                 print(f"Warning: 'normalized_ingredients' for recipe {recipe_id} is not a list or missing. Setting to empty list.")
                 recipe_dict["normalized_ingredients"] = []
            # Ensure ingredients is a list if not parsed
            if "ingredients" in recipe_dict and not isinstance(recipe_dict.get("ingredients"), list):
                 recipe_dict["ingredients"] = [] # Fallback to empty list if not parsable
            # ---> REMOVED: Do not force 'steps' to be a list if it's not <---\n",
            # if "steps" not in recipe_dict or not isinstance(recipe_dict.get("steps"), list):
            #      recipe_dict["steps"] = []
            if "n_ingredients" not in recipe_dict:
                 recipe_dict["n_ingredients"] = len(recipe_dict["ingredients"])
            if "n_steps" not in recipe_dict:
                 recipe_dict["n_steps"] = len(recipe_dict["steps"])
            # ---> END MODIFICATION <---\n",

            return json.dumps(recipe_dict, indent=2, default=str) # Use default=str for safety

    except sqlite3.Error as e:
        print(f"ERROR in get_recipe_by_id (SQL): {e}")
        return json.dumps({"error": f"Database error fetching recipe ID {recipe_id}: {e}"})
    except Exception as e:
        print(f"ERROR in get_recipe_by_id: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({"error": f"Unexpected error fetching recipe ID {recipe_id}: {e}"})

@tool
def get_ratings_and_reviews_by_recipe_id(recipe_id: str, limit: int = 5) -> str: # ---> MODIFIED: Default limit to 5 <---
    """
    Retrieves the overall average rating and the most recent reviews (up to 'limit')
    for a given recipe ID from the SQL database. Also attempts to retrieve individual ratings
    for dashboard display. Requires a positive integer for 'limit'. Returns data as a JSON string.
    """
    print(f"DEBUG TOOL CALL: get_ratings_and_reviews_by_recipe_id(recipe_id='{recipe_id}', limit={limit})\")")
    if not isinstance(recipe_id, str) or not recipe_id.isdigit():
         try:
             recipe_id_int = int(recipe_id)
             recipe_id = str(recipe_id_int)
         except (ValueError, TypeError):
              return json.dumps({"status": "error", "message": f"Invalid recipe_id format: '{recipe_id}'. Must be a numeric string."})

    try:
        limit_int = int(limit)
        if limit_int <= 0: raise ValueError("'limit' must be positive.")
    except (ValueError, TypeError):
        return json.dumps({"error": f"'limit' parameter must be a positive integer. Got: {limit}"})

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Check if recipe exists
            cursor.execute("SELECT COUNT(*) FROM recipes WHERE id = ?", (int(recipe_id),))
            if cursor.fetchone()[0] == 0:
                 return json.dumps({"status": "not_found", "message": f"Recipe ID {recipe_id} not found."})

            # Get overall average rating
            cursor.execute("SELECT AVG(rating) FROM interactions WHERE recipe_id = ?", (int(recipe_id),))
            overall_rating_result = cursor.fetchone()
            overall_rating = round(overall_rating_result[0], 2) if overall_rating_result and overall_rating_result[0] is not None else None

            # Get recent reviews (text)
            cursor.execute(
                """SELECT date, rating, review
                   FROM interactions
                   WHERE recipe_id = ? AND review IS NOT NULL AND review != ''
                   ORDER BY date DESC LIMIT ?""",
                (int(recipe_id), limit_int),
            )
            recent_reviews_raw = cursor.fetchall()
            review_columns = ["date", "rating", "review"]
            reviews_list = [dict(zip(review_columns, review)) for review in recent_reviews_raw]

            # ---> ADDED: Attempt to get individual ratings for breakdown <---
            # This assumes the 'interactions' table has individual ratings.
            # Fetch a larger sample for potentially better distribution, e.g., last 50 ratings
            cursor.execute(
                "SELECT rating FROM interactions WHERE recipe_id = ? ORDER BY date DESC LIMIT 50",
                (int(recipe_id),)
            )
            all_ratings_raw = cursor.fetchall()
            all_ratings = [r[0] for r in all_ratings_raw if r[0] is not None]
            # ---> END ADDITION <---

            result_dict = {
                "recipe_id": recipe_id,
                "overall_rating": overall_rating,
                "recent_reviews": reviews_list,
                "all_ratings_sample": all_ratings # Added for dashboard
            }
            return json.dumps(result_dict, indent=2)
    except sqlite3.Error as e:
        print(f"ERROR in get_ratings_and_reviews_by_recipe_id (SQL): {e}")
        return json.dumps({"error": f"Database error fetching reviews for recipe ID {recipe_id}: {e}"})
    except Exception as e:
        print(f"ERROR in get_ratings_and_reviews_by_recipe_id: {e}")
        return json.dumps({"error": f"Unexpected error fetching reviews for recipe ID {recipe_id}: {e}"})


@tool
def fetch_nutrition_from_usda_fdc(ingredient_name: str) -> str:
    # ... (keep existing function - no changes needed here) ...
    """
    Fetches nutrition data (per 100g) for a single ingredient from USDA FoodData Central API.
    Requires the USDA_API_KEY environment variable to be set. Includes robust retry logic.
    Returns nutrition data as a JSON string or an error/unavailable status.
    """
    print(f"DEBUG TOOL CALL: fetch_nutrition_from_usda_fdc(ingredient_name='{ingredient_name}')")
    api_key = UserSecretsClient().get_secret("USDA_API_KEY")


    if not api_key:
        print("ERROR: USDA_API_KEY environment variable not set.")
        return json.dumps({"error": "USDA FDC API key environment variable not set."})

    search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": ingredient_name,
        "api_key": api_key,
        "pageSize": 1,
        "dataType": "SR Legacy,Foundation", # Prioritize standard reference / foundation
    }
    headers = {'User-Agent': 'KitchenAssistantLangGraph/1.0 (Language: Python)'}

    max_retries = 3
    base_timeout = 15
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.get(search_url, params=params, headers=headers, timeout=base_timeout)
            response.raise_for_status()
            data = response.json()

            if data.get('foods') and len(data['foods']) > 0:
                food_item = data['foods'][0]
                fdc_nutrients = food_item.get('foodNutrients', [])

                nutrition_info = {
                    "food_normalized": ingredient_name,
                    "source": "USDA FoodData Central",
                    "product_name": food_item.get('description', ingredient_name),
                    "fdc_id": food_item.get('fdcId'),
                    "data_type": food_item.get('dataType'),
                }

                found_nutrients = {}
                for nutrient in fdc_nutrients:
                    nutrient_name = nutrient.get('nutrientName')
                    nutrient_value = nutrient.get('value') # Keep value as reported
                    target_key = FDC_NUTRIENT_MAP.get(nutrient_name)
                    if target_key:
                        found_nutrients[target_key] = nutrient_value

                nutrition_info.update(found_nutrients)
                filtered_nutrition = {k: v for k, v in nutrition_info.items() if v is not None}

                core_nutrients = ["calories_100g", "fat_100g", "proteins_100g", "carbohydrates_100g"]
                has_core_data = False
                for core_key in core_nutrients:
                    if core_key in filtered_nutrition:
                        try:
                            float(filtered_nutrition[core_key])
                            has_core_data = True
                            break
                        except (ValueError, TypeError): continue

                if not has_core_data:
                    print(f"--> No core numeric nutrition data found for '{ingredient_name}' in product '{filtered_nutrition.get('product_name', 'N/A')}' (FDC ID: {filtered_nutrition.get('fdc_id')})\")")
                    return json.dumps({"status": "unavailable", "reason": f"No detailed numeric core nutrition data found for '{ingredient_name}'"})

                print(f"--> Successfully found nutrition data for '{ingredient_name}' via USDA FDC")
                return json.dumps(filtered_nutrition, indent=2)
            else:
                print(f"--> No product found for '{ingredient_name}' via USDA FDC")
                return json.dumps({"status": "unavailable", "reason": f"No product found for '{ingredient_name}' on USDA FDC"})

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                 print(f"HTTP Error 403 (Forbidden) for '{ingredient_name}'. Check your USDA FDC API key.")
                 return json.dumps({"status": "error", "reason": f"API request failed with HTTP 403 (Forbidden). Check API Key."})
            elif e.response.status_code == 429 and attempt < max_retries - 1:
                wait_time = (retry_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"Rate limit hit for '{ingredient_name}'. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})\")")
                time.sleep(wait_time)
                continue
            elif e.response.status_code >= 500 and attempt < max_retries - 1:
                 wait_time = (retry_delay * (2 ** attempt)) + random.uniform(0, 1)
                 print(f"Server error ({e.response.status_code}) for '{ingredient_name}'. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})\")")
                 time.sleep(wait_time)
                 continue
            else:
                print(f"HTTP Error fetching nutrition for '{ingredient_name}': {e}")
                return json.dumps({"status": "unavailable", "reason": f"API request failed with HTTP error: {e}"})

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (retry_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"Request error for '{ingredient_name}': {e}. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})\")")
                time.sleep(wait_time)
                continue
            else:
                print(f"Error fetching nutrition for '{ingredient_name}' after {max_retries} attempts: {e}")
                return json.dumps({"status": "unavailable", "reason": f"API request failed after retries: {e}"})

        except json.JSONDecodeError:
            print(f"Error decoding JSON response for '{ingredient_name}'")
            return json.dumps({"status": "unavailable", "reason": "Invalid JSON response from API"})

        except Exception as e:
             print(f"ERROR in fetch_nutrition_from_usda_fdc: {e}")
             import traceback
             traceback.print_exc()
             return json.dumps({"error": f"Unexpected error fetching nutrition for {ingredient_name}: {e}"})

    print(f"Max retries ({max_retries}) exceeded for API request for '{ingredient_name}'")
    return json.dumps({"status": "unavailable", "reason": f"Max retries ({max_retries}) exceeded for API request for '{ingredient_name}'"})


# ---> REVISED: customize_recipe Tool (Slightly Improved Placeholder Logic) <---\n",
import json
import re # Make sure re is imported if not already done earlier in the cell
from typing import Optional # Ensure Optional is imported

@tool
def customize_recipe(recipe_id: str, request: str, recipe_details_json: Optional[str] = None) -> str:
    """
    Attempts to customize a recipe based on a user request (e.g., make vegetarian, substitute ingredient, make gluten-free, make low-fat/healthier).
    Requires the recipe_id and the specific customization request string.
    Optionally takes current recipe_details as JSON string to avoid re-fetching.
    Returns a JSON string describing the suggested modifications or indicating inability.
    """
    print(f"DEBUG TOOL CALL: customize_recipe(recipe_id='{recipe_id}', request='{request}')")
    if not recipe_id or not request:
        return json.dumps({"status": "error", "message": "Missing recipe_id or customization request."})
    if not isinstance(recipe_id, str) or not recipe_id.isdigit():
         try:
             recipe_id_int = int(recipe_id)
             recipe_id = str(recipe_id_int)
         except (ValueError, TypeError):
              return json.dumps({"status": "error", "message": f"Invalid recipe_id format: '{recipe_id}'. Must be numeric string."})

    # --- Placeholder Logic (Slightly Enhanced) ---
    # In a real implementation: Parse details, analyze request (NLP/LLM), lookup substitutions, generate specific steps.

    request_lower = request.lower()
    modifications = []
    status = "placeholder_success"
    message_lines = [f"Suggestions for making recipe {recipe_id} '{request}':"]

    # Basic Keyword Checks
    if "vegan" in request_lower:
        modifications.append({"action": "replace", "original": "dairy/meat/eggs", "suggestion": "plant-based alternatives (e.g., tofu, plant milk, oil instead of butter, nutritional yeast for cheese)"})
        message_lines.append("- Check all ingredients for animal products (stock, cheese, etc.) and replace with vegan versions.")
    elif "gluten-free" in request_lower or "gluten free" in request_lower:
        modifications.append({"action": "replace", "original": "wheat flour/pasta/soy sauce", "suggestion": "certified gluten-free alternatives (GF flour blend, GF pasta, tamari)"})
        message_lines.append("- Replace any gluten-containing grains (wheat, barley, rye) with gluten-free options.")
    elif "low fat" in request_lower or "low-fat" in request_lower or "healthy" in request_lower or "healthier" in request_lower:
        modifications.append({"action": "reduce/replace", "original": "high-fat ingredients (e.g., butter, cream, fatty meats, cheese spread)", "suggestion": "lower-fat options (e.g., olive oil sparingly, low-fat milk/yogurt, lean protein, reduced-fat cheese)"})
        message_lines.append("- Reduce added fats like butter or oil where possible.")
        message_lines.append("- Use low-fat dairy alternatives.")
        message_lines.append("- Consider adding more vegetables for volume and nutrients.")
        # Simple check if details provided and contain the ingredient
        if recipe_details_json and "american cheese spread" in recipe_details_json:
             message_lines.append("- Specifically, consider reducing or replacing the American cheese spread with a lower-fat option or nutritional yeast for cheesy flavor.")
    elif "substitute" in request_lower or "replace" in request_lower:
        # Use regex to find "substitute X for/with Y" or "replace X for/with Y"
        match = re.search(r"(?:substitute|replace)\s+(.*?)\s+(?:for|with)\s+(.*)", request_lower)
        original = "ingredient_A"
        suggestion = "ingredient_B"
        if match:
            original = match.group(1).strip()
            suggestion = match.group(2).strip()
        modifications.append({"action": "replace", "original": original, "suggestion": suggestion})
        message_lines.append(f"- You could try replacing '{original}' with '{suggestion}'. Check quantities and cooking times.")
    else:
        # If no specific keywords match, provide a generic placeholder or ask for clarification
        status = "placeholder_needs_clarification" # Changed status
        message_lines.append(f"- I can provide general suggestions for '{request}'. Review ingredients and consider reducing fats/sugars or increasing vegetables.")
        message_lines.append("- For specific substitutions (like 'replace X with Y'), please state them clearly.")
        modifications.append({"action": "general_advice", "original": request, "suggestion": "Review recipe based on request"})

    # Add a default message if no specific modifications were generated but status is success
    if status == "placeholder_success" and not modifications:
         message_lines.append("- No specific substitutions identified by this placeholder logic, but review ingredients based on your request.")

    return json.dumps({
        "status": status,
        "message": "\n".join(message_lines), # Join lines for better formatting in the final message
        "recipe_id": recipe_id,
        "original_request": request,
        "suggested_modifications": modifications
    })
# ---> END REVISION <---\n",

# ---> ADDED: fetch_live_recipe_data Tool (Placeholder) <---
@tool
def fetch_live_recipe_data(recipe_id: str) -> str:
    """
    (Placeholder) Attempts to fetch live recipe data (ingredients, steps, time)
    from food.com using the recipe ID. Returns data as a JSON string or an error status.
    Requires requests and beautifulsoup4 libraries.
    """
    print(f"DEBUG TOOL CALL: fetch_live_recipe_data(recipe_id='{recipe_id}')")
    if not isinstance(recipe_id, str) or not recipe_id.isdigit():
        return json.dumps({"status": "error", "message": "Invalid recipe_id format."})

    url = f"https://www.food.com/recipe/{recipe_id}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'} # Basic user agent

    try:
        # --- Placeholder Scraping Logic ---
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Example: Extracting ingredients (CSS selectors WILL change)
        ingredients_elements = soup.select('.recipe-ingredients__list .recipe-ingredients__item-label')
        ingredients = [el.get_text(strip=True) for el in ingredients_elements]

        # Example: Extracting time (CSS selectors WILL change)
        time_element = soup.select_one('.recipe-facts__details--total-time .value')
        total_time_str = time_element.get_text(strip=True) if time_element else None
        # Parse total_time_str into minutes (e.g., using regex)
        total_minutes = None # Placeholder for parsed time

        # ... extract other fields like steps, description ...

        if not ingredients: # Basic check if scraping failed
            raise ValueError("Could not extract ingredients from page.")

        live_data = {
            "source": "food.com (live)",
            "url": url,
            "ingredients": ingredients,
            "minutes": total_minutes,
            # "steps": extracted_steps,
            # "description": extracted_description,
        }
        # return json.dumps({"status": "live_success", "data": live_data})
        # --- End Placeholder ---

        # Return placeholder success for now
        print(f"Placeholder: Would attempt to scrape {url}")
        return json.dumps({
            "status": "placeholder_success",
            "message": f"Placeholder success for fetching live data for recipe {recipe_id}.",
            "data": {
                "source": "food.com (placeholder)",
                "url": url,
                "ingredients": ["Placeholder Ingredient 1", "Placeholder Ingredient 2"],
                "minutes": 45, # Placeholder value
                "steps": ["Placeholder Step 1", "Placeholder Step 2"],
                "description": "Placeholder live description."
            }
        })

    except Exception as e:
        print(f"ERROR in fetch_live_recipe_data for recipe {recipe_id}: {e}")
        return json.dumps({"status": "error", "message": f"Failed to fetch or parse live data: {e}"})
# ---> END ADDITION <---


# --- Nutrition Visualization Function ---
# ---> MODIFIED: Add header_constant argument <---
# In cell 5ead34ad-8b23-4018-a4d7-da4a854eebce
# Find the --- Nutrition Visualization Function --- section
# --- Nutrition Visualization Function ---
# ---> MODIFIED: Add header_constant argument <---\n",
import re
import matplotlib.pyplot as plt
from typing import Dict

def extract_and_visualize_nutrition(response_text: str, header_constant: str):
    """Extracts nutrition data from text starting with header_constant and plots it."""
    print("Attempting to extract and visualize nutrition...")

    if not isinstance(response_text, str):
        print(f"Error: Expected string input for visualization, got {type(response_text)}")
        return # Cannot proceed if input isn't a string

    if not header_constant:
        print("ERROR: No header constant provided for visualization.")
        return

    try:
        header_pattern = re.escape(header_constant)
    except Exception as e:
        print(f"ERROR creating regex pattern from header: {e}")
        return

    nutrition_section_match = re.search(
        rf"{header_pattern}.*?:\s*\n(.*?)(?:$|\n\s*\n|\n\(Note:|\Z)", # Slightly adjusted pattern end
        response_text,
        re.DOTALL | re.IGNORECASE
    )

    if not nutrition_section_match:
        print(f"Could not find the nutrition section starting with '{header_constant}' in the text.")
        # ---> ADDED: Print the text it searched in for debugging <---
        print(f"Searched text (first 300 chars): {response_text[:300]}...")
        return

    nutrition_text = nutrition_section_match.group(1).strip()
    print(f"Extracted Nutrition Text Block:\n---\n{nutrition_text}\n---")

    nutrient_pattern = re.compile(
        r"^\s*-\s*(?P<nutrient>[^:]+?)\s*:\s*(?P<value>[\d.]+)\s*(?P<unit>kcal|g|mg).*",
        re.MULTILINE | re.IGNORECASE
    )

    # ... (rest of the key_map, extraction logic remains the same) ...
    key_map = {
        'calories': 'calories_100g',
        'fat': 'fat_100g',
        'saturated fat': 'saturated_fat_100g',
        'carbohydrates': 'carbohydrates_100g',
        'sugars': 'sugars_100g',
        'fiber': 'fiber_100g',
        'proteins': 'proteins_100g',
        'sodium': 'sodium_100g'
    }

    extracted_values: Dict[str, float] = {}
    extracted_units: Dict[str, str] = {}
    processed_nutrients = 0

    for match in nutrient_pattern.finditer(nutrition_text):
        nutrient_name = match.group("nutrient").strip().lower()
        value_str = match.group("value").strip()
        unit = match.group("unit").strip().lower()

        if nutrient_name in key_map:
            state_key = key_map[nutrient_name]
            try:
                value = float(value_str)
                extracted_values[state_key] = value
                extracted_units[state_key] = unit
                processed_nutrients += 1
                print(f"Extracted: {state_key} = {value} {unit}")
            except ValueError:
                print(f"Warning: Could not convert value '{value_str}' for '{nutrient_name}'.")
        else:
            print(f"Warning: Unrecognized nutrient '{nutrient_name}'.")

    if processed_nutrients == 0:
        print("No valid nutrition data found to plot.")
        return

    print(f"Processed {processed_nutrients} nutrients.")
    print("Values:", extracted_values)
    print("Units:", extracted_units)


    # ... (DV calculation and plotting logic remains the same) ...
    # ... (Make sure plt is imported correctly) ...
    daily_values = {
        "calories_100g": 2000, "fat_100g": 78, "saturated_fat_100g": 20,
        "carbohydrates_100g": 275, "sugars_100g": 50, "fiber_100g": 28,
        "proteins_100g": 50, "sodium_100g": 2300
    }
    percent_dv: Dict[str, float] = {}
    actual_values_plot: Dict[str, float] = {}

    for key, value in extracted_values.items():
        dv = daily_values.get(key)
        unit = extracted_units.get(key, 'g')
        if dv is not None and dv > 0:
            value_for_calc = value
            if key == 'sodium_100g' and unit == 'g': value_for_calc *= 1000.0
            elif unit == 'mg' and key != 'sodium_100g': value_for_calc /= 1000.0

            percent_dv[key] = round((value_for_calc / dv) * 100, 1)
            actual_values_plot[key] = round(value, 1)
        else:
            percent_dv[key] = 0.0
            actual_values_plot[key] = round(value, 1)

    calories_percent_dv = percent_dv.pop("calories_100g", 0.0)
    calories_actual = actual_values_plot.pop("calories_100g", 0.0)
    # ---> Filter only non-zero %DV AND ensure key exists in actual_values_plot <---
    plot_data = {k: v for k, v in percent_dv.items() if k in actual_values_plot and v > 0}

    if not plot_data:
        print("No data with %DV > 0 to plot.")
        if calories_actual > 0:
            print(f"Avg Calories: {calories_actual:.0f} kcal")
        return

    labels = list(plot_data.keys())
    display_labels = [l.replace('_100g', '').replace('_', ' ').capitalize() for l in labels]
    values = list(plot_data.values())
    colors = ['forestgreen' if v <= 15 else ('orange' if v <= 40 else 'red') for v in values]

    fig = None # Initialize fig to None
    try:
        # ---> Use subplots for better figure management <---
        fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.6)))
        bars = ax.barh(display_labels, values, color=colors, height=0.6)
        ax.set_xlabel('% Daily Value (DV) - Based on average of 100g of each ingredient')
        ax.set_title('Average Ingredient Nutrition (%DV)', fontsize=16)
        ax.tick_params(axis='both', labelsize=10)
        max_val = max(values + [100]) # Ensure max_val includes 100 for percentage scale
        ax.set_xlim(right=max_val * 1.1) # Adjust xlim dynamically

        for i, bar in enumerate(bars):
            width = bar.get_width()
            nutrient_key = labels[i]
            actual_val = actual_values_plot.get(nutrient_key, 0.0)
            unit = extracted_units.get(nutrient_key, 'g')
            if nutrient_key == 'sodium_100g': unit = 'mg'
            display_actual = f"{actual_val:.1f}"
            label_text = f'{width:.1f}% ({display_actual} {unit})'
            # Dynamic text positioning
            x_pos = width + max_val * 0.01 if width < max_val * 0.85 else width - max_val * 0.01
            ha = 'left' if width < max_val * 0.85 else 'right'
            color = 'black' if width < max_val * 0.85 else 'white'
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2., label_text,
                    ha=ha, va='center', color=color, fontsize=9, fontweight='bold')

        cal_color = 'forestgreen' if calories_percent_dv <= 15 else ('orange' if calories_percent_dv <= 40 else 'red')
        calorie_text = f'Estimated Avg Calories per 100g Ingredient: {calories_actual:.0f} kcal ({calories_percent_dv:.1f}% DV)'
        fig.text(0.5, 0.97, calorie_text, ha='center', va='bottom', fontsize=12, color=cal_color, fontweight='bold')

        plt.gca().invert_yaxis() # Keep high %DV at top
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ---> Crucial for Kaggle/non-interactive: Ensure plot is shown <---
        plt.show()
        print("Nutrition visualization displayed.")

    except Exception as plot_error:
        print(f"Error during plotting: {plot_error}")
        # ---> ADDED: Close the figure if it was created but failed during plot generation <---
        if fig:
            plt.close(fig)



# --- Tool Lists & Executor ---
stateless_tools = [ gemini_recipe_similarity_search, get_recipe_by_id, get_ratings_and_reviews_by_recipe_id, fetch_nutrition_from_usda_fdc, fetch_live_recipe_data, customize_recipe ]
llm_callable_tools = stateless_tools + [customize_recipe]
tool_executor_node = ToolNode(stateless_tools)

# --- LLM Binding ---
# llm should be defined in Step 2
llm_with_callable_tools = llm.bind_tools(llm_callable_tools)



print("✅ LangGraph Step 3: Tools Defined and Bound (Revised)")


# ## 10.4. The Decision Maker (Input Parser Node)
# 
# The `input_parser_node` acts as the central hub after receiving input (either from the user or from a tool's output).
# 
# *   **Intelligent Routing:** Its primary job is to analyze the current state and the latest message. Instead of *always* asking the LLM what to do next, it now has logic to *directly route* to specific processing nodes if it receives results from certain tools (like `get_recipe_by_id`, `get_ratings_and_reviews_by_recipe_id`, `fetch_nutrition_from_usda_fdc` batch, `customize_recipe`, `fetch_live_recipe_data`). This makes the agent more efficient.
# *   **LLM Invocation:** If direct routing isn't applicable (e.g., initial user input, ambiguous request, general chat), it invokes the LLM (with tools bound) to decide the next action: call a tool, ask for clarification, provide a chat response, or end the conversation.
# *   **Context Management:** It helps manage the conversation context, like identifying the `selected_recipe_id`.

# In[45]:


# LangGraph Step 4: Core Node Definitions (Revised Input Parser for Recipe Details)

# --- Assume KitchenState, KITCHEN_ASSISTANT_SYSINT, llm_with_callable_tools are defined ---

def input_parser_node(state: KitchenState) -> Dict[str, Any]:
    """
    Parses user input or tool results. Routes directly after processing key tool results
    (like recipe details, reviews, customization) to avoid unnecessary LLM calls.
    Otherwise, uses the LLM to determine the next step.
    """
    print("---NODE: InputParserNode---")
    messages = state['messages']
    last_message = messages[-1] if messages else None
    previous_ai_message: Optional[AIMessage] = None
    for i in range(len(messages) - 2, -1, -1):
        if isinstance(messages[i], AIMessage):
            previous_ai_message = messages[i]
            break

    # --- Direct Routing based on incoming ToolMessage ---

    # A. Nutrition Aggregation Check
    needs_aggregation = False
    last_message = messages[-1] if messages else None

    # ---> Refined Check: Only trigger aggregation if the LAST message is a nutrition tool message <---
    # ---> AND if we can trace back to an AI message that called multiple nutrition tools <---
    if isinstance(last_message, ToolMessage) and last_message.name == "fetch_nutrition_from_usda_fdc":
        print("DEBUG: Last message is a nutrition ToolMessage.")
        # Find the AIMessage that likely requested this batch
        last_ai_request_index = -1
        for i in range(len(messages) - 2, -1, -1): # Search backwards from message before last
            msg = messages[i]
            if isinstance(msg, AIMessage) and msg.tool_calls:
                nutrition_call_count = sum(1 for tc in msg.tool_calls if tc.get('name') == 'fetch_nutrition_from_usda_fdc')
                if nutrition_call_count > 1: # Check if *multiple* calls were made
                    print(f"DEBUG: Found preceding AIMessage at index {i} with {nutrition_call_count} nutrition tool calls.")
                    last_ai_request_index = i
                    break
                elif nutrition_call_count == 1:
                    print(f"DEBUG: Found preceding AIMessage at index {i} with only 1 nutrition tool call. Not aggregating.")
                    break # Stop searching if only one was called
            # Stop if we hit a human message before finding a relevant AI message
            if isinstance(msg, HumanMessage):
                print("DEBUG: Hit HumanMessage before finding multi-nutrition AI request.")
                break

        # Check if all messages after the AI request (up to the current one) are nutrition tool messages
        if last_ai_request_index != -1:
            all_nutrition_results_in_batch = True
            # Only check messages between the AI request and the current message
            for i in range(last_ai_request_index + 1, len(messages)):
                msg = messages[i]
                if not (isinstance(msg, ToolMessage) and msg.name == "fetch_nutrition_from_usda_fdc"):
                    print(f"DEBUG: Found non-nutrition ToolMessage ({type(msg).__name__}, name={getattr(msg, 'name', 'N/A')}) in batch. Not aggregating yet.")
                    all_nutrition_results_in_batch = False
                    break
            if all_nutrition_results_in_batch:
                needs_aggregation = True
                print("DEBUG: Detected likely end of nutrition tool results batch.")
        else:
            print("DEBUG: No preceding AI message found requesting multiple nutrition lookups.")


    if needs_aggregation:
        print("Routing to aggregation.")
        # Ensure necessary context is preserved for aggregation/formatting
        updates = {
            "intent": "aggregate_nutrition",
            "messages": [], # Clear messages for the aggregation node step
            "selected_recipe_id": state.get("selected_recipe_id"),
            "current_recipe_details": state.get("current_recipe_details") # Pass details for recipe name
            }
        valid_keys = KitchenState.__annotations__.keys()
        return {k: v for k, v in updates.items() if k in valid_keys}

    # B. Review Processing Check
    if isinstance(last_message, ToolMessage) and last_message.name == "get_ratings_and_reviews_by_recipe_id":
         print("Detected review tool results, setting intent to process reviews.")
         try: review_content = json.loads(last_message.content)
         except: review_content = {"error": "Failed to parse review tool content"}
         updates = { "intent": "process_reviews", "messages": [], "recipe_reviews": review_content, "selected_recipe_id": state.get("selected_recipe_id"), "current_recipe_details": state.get("current_recipe_details") }
         print(f"Routing to review processing.")
         return {k: v for k, v in updates.items() if k in KitchenState.__annotations__}

    # C. Customization Processing Check
    if isinstance(last_message, ToolMessage) and last_message.name == "customize_recipe":
         print("Detected customization tool results, setting intent to process customization.")
         updates = { "intent": "process_customization", "messages": [], "selected_recipe_id": state.get("selected_recipe_id"), "current_recipe_details": state.get("current_recipe_details") }
         print(f"Routing to customization processing.")
         return {k: v for k, v in updates.items() if k in KitchenState.__annotations__}

    # D. Live Data Check (Route to Formatter)
    if isinstance(last_message, ToolMessage) and last_message.name == "fetch_live_recipe_data":
         print("Detected live recipe data results.")
         try: live_content = json.loads(last_message.content)
         except: live_content = {"error": "Failed to parse live data tool content"}
         updates = { "intent": "live_data_fetched", "messages": [], "live_recipe_details": live_content, "selected_recipe_id": state.get("selected_recipe_id"), "current_recipe_details": state.get("current_recipe_details") }
         print(f"Routing to formatter after fetching live data.")
         return {k: v for k, v in updates.items() if k in KitchenState.__annotations__}

    # ---> ADDED: Recipe Details Check (Route to Formatter) <---
    if isinstance(last_message, ToolMessage) and last_message.name == "get_recipe_by_id":
        print("Detected recipe details tool results, updating state and routing to formatter.")
        try:
            details_content = json.loads(last_message.content)
            # Check if the tool returned an error or not found status
            if isinstance(details_content, dict) and details_content.get("status") in ["error", "not_found"]:
                 # If tool failed, let LLM handle the error message
                 print("Recipe details tool returned error/not_found, proceeding with LLM.")
                 pass # Fall through to LLM invocation below
            elif isinstance(details_content, dict):
                 # Success! Update state and route to formatter
                 updates = {
                     "intent": "recipe_details_fetched", # Signal for formatter
                     "messages": [], # Prevent re-processing by LLM
                     "current_recipe_details": details_content, # Store the fetched details
                     "selected_recipe_id": state.get("selected_recipe_id"), # Keep context
                     # Clear potentially stale related data
                     "live_recipe_details": None,
                     "processed_review_data": None,
                     "customization_results": None,
                     "recipe_reviews": None,
                 }
                 print(f"Routing to formatter after fetching recipe details.")
                 return {k: v for k, v in updates.items() if k in KitchenState.__annotations__}
            else:
                 # Unexpected content format, let LLM handle
                 print("Unexpected format for recipe details tool result, proceeding with LLM.")
                 pass # Fall through
        except Exception as e:
            print(f"Error processing recipe details tool message: {e}. Proceeding with LLM.")
            pass # Fall through to LLM invocation on error

    # ---> END ADDITION <---


    # --- Normal LLM Invocation (If no direct routing occurred) ---
    # This handles: Initial user input, interpreting other tool results, general chat, errors from previous steps
    print("Proceeding with LLM invocation...")
    context_messages = [SystemMessage(content=KITCHEN_ASSISTANT_SYSINT[1])] + list(messages)
    try:
        ai_response: AIMessage = llm_with_callable_tools.invoke(context_messages)
        print(f"LLM Raw Response: {ai_response}")
    except Exception as e:
        print(f"LLM Invocation Error: {e}")
        error_message = "Sorry, I encountered an internal error trying to process that. Could you try rephrasing?"
        return { "messages": [AIMessage(content=error_message)], "last_assistant_response": error_message, "intent": "error", "finished": False, "selected_recipe_id": state.get("selected_recipe_id"), "current_recipe_details": state.get("current_recipe_details") }

    # Prepare state updates based on LLM response
    updates = {
        "messages": [ai_response],
        "intent": "general_chat", # Default intent
        "finished": False,
        "last_assistant_response": None,
        "needs_clarification": False,
        # Preserve context
        "selected_recipe_id": state.get("selected_recipe_id"),
        "current_recipe_details": state.get("current_recipe_details"),
        "recipe_reviews": state.get("recipe_reviews"),
        "live_recipe_details": state.get("live_recipe_details"),
        # Clear transient fields
        "ingredient_nutrition_list": None,
        "nutritional_info": None,
        "processed_review_data": None,
        "customization_results": None,
        "grounding_results_formatted": None,
        "customization_request": None,
    }

    if ai_response.tool_calls:
        updates["intent"] = "tool_call" # General intent for routing to executor
        print(f"Intent: tool_call, Tool Calls: {ai_response.tool_calls}\")")

        # Context Management & Intent Setting based on Tool Calls
        # ... (Keep the existing context management logic here) ...
        new_search_initiated = any(tc.get('name') == 'gemini_recipe_similarity_search' for tc in ai_response.tool_calls)
        if new_search_initiated:
            print("New recipe search detected, clearing previous recipe context.")
            updates["current_recipe_details"] = None; updates["selected_recipe_id"] = None; updates["recipe_reviews"] = None; updates["nutritional_info"] = None; updates["live_recipe_details"] = None; updates["processed_review_data"] = None; updates["customization_results"] = None

        for tc in ai_response.tool_calls:
            tool_name = tc.get('name'); tool_args = tc.get('args', {}); recipe_id_arg = tool_args.get('recipe_id')
            if tool_name in ['get_recipe_by_id', 'get_ratings_and_reviews_by_recipe_id', 'customize_recipe', 'fetch_live_recipe_data'] and recipe_id_arg:
                if recipe_id_arg != updates["selected_recipe_id"]:
                     print(f"Tool call for new recipe ID '{recipe_id_arg}', updating context.")
                     updates["selected_recipe_id"] = recipe_id_arg; updates["current_recipe_details"] = None; updates["recipe_reviews"] = None; updates["nutritional_info"] = None; updates["live_recipe_details"] = None; updates["processed_review_data"] = None; updates["customization_results"] = None
                updates["selected_recipe_id"] = recipe_id_arg # Ensure ID is set

            # Set specific intents to guide routing *after* tool execution
            # Note: get_recipe_by_id intent is now set above when processing the ToolMessage result
            if tool_name == 'get_ratings_and_reviews_by_recipe_id': updates["intent"] = "reviews_fetched"
            elif tool_name == 'customize_recipe':
                 updates["intent"] = "customization_requested"
                 updates["customization_request"] = tool_args.get('request')
                 if state.get("current_recipe_details"):
                     try: tool_args["recipe_details_json"] = json.dumps(state["current_recipe_details"])
                     except Exception: print("Warning: Could not serialize details for customize_recipe tool.")
            elif tool_name == 'fetch_live_recipe_data': updates["intent"] = "live_data_requested"
            elif tool_name == 'fetch_nutrition_from_usda_fdc' and not any(t['name'] == 'fetch_nutrition_from_usda_fdc' for t in ai_response.tool_calls if t != tc): # Single nutrition call
                 updates["intent"] = "single_nutrition_fetched" # Let formatter handle simple display

    elif ai_response.content:
        # Handle direct text responses from LLM
        updates["last_assistant_response"] = ai_response.content
        content_lower = ai_response.content.lower()
        # ... (keep existing intent logic for text responses: clarification, exit, general_chat) ...
        if "need more details" in content_lower or "could you clarify" in content_lower or "which recipe" in content_lower: updates["intent"] = "clarification_needed"; updates["needs_clarification"] = True
        elif "goodbye" in content_lower or "exit" in content_lower or "bye" in content_lower: updates["intent"] = "exit"; updates["finished"] = True
        elif state.get("user_input", "").lower() in {"q", "quit", "exit", "goodbye"}: updates["intent"] = "exit"; updates["finished"] = True
        else: updates["intent"] = "general_chat"
        print(f"Intent: {updates['intent']}, Response: {updates['last_assistant_response'][:100]}...")

    else: # Handle LLM error or empty response
        updates["intent"] = "error"
        error_message = "Sorry, I had trouble processing that request. Can you please try again?"
        updates["last_assistant_response"] = error_message
        updates["messages"] = [AIMessage(content=error_message)]
        print(f"Intent: error (Empty LLM response)")

    valid_keys = KitchenState.__annotations__.keys()
    return {k: v for k, v in updates.items() if k in valid_keys and (k == 'messages' or state.get(k) != v)}


print("✅ LangGraph Step 4: Core Nodes Defined (Revised Input Parser for Recipe Details)")


# ## 10.5. Define Custom Action Nodes
# ### LangGraph Step 5: Specialized Processors (Action Nodes)
# 
# While the `ToolExecutorNode` simply runs the tools, these custom nodes perform specific *processing* on the raw tool outputs *before* the final response is formatted.
# 
# *   `aggregate_nutrition_node`: Takes multiple individual ingredient nutrition results (from `fetch_nutrition_from_usda_fdc`) and calculates the average nutritional profile for the recipe.
# *   `review_dashboard_node`: Processes the raw ratings and reviews, performs sentiment analysis using VADER, calculates rating distributions, and prepares the data for the review dashboard.
# *   `process_customization_node`: Takes the output from the `customize_recipe` tool and prepares the suggestions for display.
# *   `visualize_nutrition_node`: Checks the *final formatted response* and, if it contains the nutrition summary, generates and displays the Matplotlib bar chart.
# 

# In[46]:


# LangGraph Step 5: Specific Action Nodes (Revised - Restoring Missing Nodes)

# --- Assume KitchenState, tools, constants, sentiment analyzer are defined ---
# from step1_state import KitchenState (Revised)
# from step3_tools import extract_and_visualize_nutrition, analyzer # (sentiment analyzer)
# from step2_core import NUTRITION_RESPONSE_HEADER, REVIEW_DASHBOARD_HEADER, CUSTOMIZATION_HEADER (Revised)

# --- Custom Action Nodes ---

# ---> RESTORED: aggregate_nutrition_node (from original file) <---
# Nutrition Aggregation Node (Revised for Robustness & Debugging in original file)
def aggregate_nutrition_node(state: KitchenState) -> Dict[str, Any]:
    """
    Aggregates nutrition data collected from fetch_nutrition_from_usda_fdc tool calls
    since the last AI message that requested them. Calculates average values per 100g.
    Updates the nutritional_info field in the state.
    """
    print("---NODE: AggregateNutritionNode---")
    messages = state.get("messages", [])
    aggregated_sums: Dict[str, float] = defaultdict(float) # Use defaultdict
    nutrient_counts: Dict[str, int] = defaultdict(int)
    processed_ingredient_count = 0
    unavailable_count = 0
    error_count = 0 # Add error counter
    relevant_tool_messages = []

    # Find the last AI message that made nutrition tool calls
    last_nutrition_request_index = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if any(tc.get('name') == 'fetch_nutrition_from_usda_fdc' for tc in msg.tool_calls):
                last_nutrition_request_index = i
                break
        # Stop searching if we hit the previous human message without finding an AI request
        if isinstance(msg, HumanMessage):
            break

    # Collect all ToolMessages after that specific AI request
    if last_nutrition_request_index != -1:
        start_index = last_nutrition_request_index + 1
        for i in range(start_index, len(messages)):
            msg = messages[i]
            if isinstance(msg, ToolMessage) and msg.name == "fetch_nutrition_from_usda_fdc":
                relevant_tool_messages.append(msg)
            # Stop collecting if we hit the next AI or Human message
            elif isinstance(msg, (AIMessage, HumanMessage)):
                 break
    else:
        # Fallback: Look for any nutrition tool messages in the current batch.
        print("Warning: Could not find a preceding AI message requesting nutrition.")
        print("Scanning current message list for nutrition tool results...")
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "fetch_nutrition_from_usda_fdc":
                 relevant_tool_messages.append(msg)
        if not relevant_tool_messages:
             print("No nutrition tool messages found at all.")
             return {"nutritional_info": {"processed_ingredient_count": 0, "nutrient_counts": {}}}


    print(f"Found {len(relevant_tool_messages)} relevant nutrition ToolMessages to aggregate.")
    if not relevant_tool_messages:
         print("No relevant tool messages found to aggregate.")
         return {"nutritional_info": {"processed_ingredient_count": 0, "nutrient_counts": {}}}


    # Process the relevant messages
    for msg in relevant_tool_messages:
        print(f"\nDEBUG Aggregation: Processing ToolMessage ID {getattr(msg, 'tool_call_id', 'N/A')}")
        print(f"DEBUG Aggregation: Raw Content: {msg.content}")
        try:
            content_str = msg.content
            if not isinstance(content_str, str):
                 content_str = json.dumps(content_str)

            content_data = json.loads(content_str)
            ingredient_name = content_data.get("food_normalized", "Unknown Ingredient")

            if content_data.get("status") == "unavailable":
                unavailable_count += 1
                print(f"--> Skipping unavailable result for '{ingredient_name}': {content_data.get('reason', 'No reason provided')}")
                continue
            elif "error" in content_data:
                error_count += 1
                print(f"--> Skipping error result for '{ingredient_name}': {content_data.get('error', 'Unknown error')}")
                continue

            core_nutrients = ["calories_100g", "fat_100g", "proteins_100g", "carbohydrates_100g"]
            has_core_data = False
            numeric_values_found = {}

            for key in content_data.keys():
                if key.endswith("_100g") and key not in ["food_normalized", "source", "product_name", "status", "reason", "error", "fdc_id", "data_type"]:
                    value = content_data.get(key)
                    if value is not None:
                        try:
                            num_value = float(value)
                            if num_value >= 0:
                                aggregated_sums[key] += num_value
                                nutrient_counts[key] += 1
                                numeric_values_found[key] = num_value
                                if key in core_nutrients:
                                    has_core_data = True
                            else:
                                print(f"--> Warning: Ignoring negative value '{num_value}' for key '{key}' in '{ingredient_name}'.")
                        except (ValueError, TypeError):
                            print(f"--> Warning: Could not convert value '{value}' for key '{key}' in '{ingredient_name}' to float.")

            if not has_core_data:
                unavailable_count += 1
                print(f"--> Skipping result for '{ingredient_name}': Parsed OK, but no core numeric nutrition data found. Found: {numeric_values_found}")
                continue
            else:
                processed_ingredient_count += 1
                print(f"--> Successfully processed nutrition for: {ingredient_name}")

        except json.JSONDecodeError:
            error_count += 1
            print(f"--> ERROR: Could not parse ToolMessage content as JSON: {str(msg.content)[:100]}...")
        except Exception as e:
             error_count += 1
             print(f"--> ERROR: Unexpected error processing ToolMessage ({getattr(msg, 'tool_call_id', 'N/A')}): {e}")
             import traceback
             traceback.print_exc()

    # Calculate averages
    average_nutrition = {}
    for key, total_sum in aggregated_sums.items():
        count = nutrient_counts[key]
        average_nutrition[key] = round(total_sum / count, 2) if count > 0 else 0.0

    average_nutrition["processed_ingredient_count"] = processed_ingredient_count
    average_nutrition["unavailable_ingredient_count"] = unavailable_count
    average_nutrition["error_ingredient_count"] = error_count
    average_nutrition["nutrient_counts"] = dict(nutrient_counts)

    print(f"\nAggregation Complete. Processed: {processed_ingredient_count}, Unavailable/NoData: {unavailable_count}, Errors: {error_count}")
    print(f"Aggregated Nutrition (Avg per 100g): {average_nutrition}")

    # Update state
    return {"nutritional_info": average_nutrition, "ingredient_nutrition_list": None} # Clear temp list
# ---> END RESTORED <---


# ---> RESTORED: visualize_nutrition_node (from original file) <---
# Visualization Node (Revised to CALL the function in original file)
# ---> MODIFIED: visualize_nutrition_node to pass the header <---
def visualize_nutrition_node(state: KitchenState) -> Dict[str, Any]:
    """
    Calls the nutrition visualization function using the final assistant response
    if it contains the expected nutrition information header. Passes the header constant.
    """
    print("---NODE: VisualizeNutritionNode---")
    final_response = state.get("last_assistant_response")

    # Ensure NUTRITION_RESPONSE_HEADER is accessible (defined in Step 2)
    try:
        header_check = NUTRITION_RESPONSE_HEADER
    except NameError:
        print("Warning: NUTRITION_RESPONSE_HEADER not found for visualization check.")
        header_check = None # Set to None if not found

    if final_response and header_check and header_check in final_response: # Check header exists and is in response
        print("Detected nutrition info in final response. Calling visualization function.")
        try:
            # Call the visualization function defined/imported (e.g., from Step 3)
            # Ensure extract_and_visualize_nutrition is defined or imported
            # ---> Pass the header constant <---
            extract_and_visualize_nutrition(final_response, header_check)
            print("Visualization function executed.")
        except NameError:
             print("ERROR: extract_and_visualize_nutrition function not found.")
        except Exception as e:
            print(f"Error during visualization call: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for plotting errors
    else:
        if not header_check:
             print("Skipping visualization because NUTRITION_RESPONSE_HEADER is not defined.")
        else:
             print(f"No nutrition section header ('{header_check}') found in the final response, skipping visualization.")

    return {} # This node primarily performs a side effect
# ---> END MODIFICATION <---



# ---> ADDED: ReviewDashboardNode (from previous response) <---
def review_dashboard_node(state: KitchenState) -> Dict[str, Any]:
    """
    Processes raw review data from the state ('recipe_reviews'), performs sentiment analysis (if available),
    calculates rating breakdown, selects reviews to display, and stores the processed data
    in 'processed_review_data' for the formatter node.
    """
    print("---NODE: ReviewDashboardNode---")
    raw_review_data = state.get("recipe_reviews") # This should come from get_ratings_and_reviews tool output

    if not raw_review_data or not isinstance(raw_review_data, dict):
        print("No raw review data found in state.")
        return {"processed_review_data": None, "intent": "reviews_processed_nodata"} # Signal no data

    processed_data = {
        "recipe_id": raw_review_data.get("recipe_id"),
        "overall_rating": raw_review_data.get("overall_rating"),
        "rating_counts": {},
        "sentiment_scores": {'positive': 0, 'negative': 0, 'neutral': 0},
        "reviews_for_display": []
    }

    # Calculate rating breakdown from the sample
    all_ratings = raw_review_data.get("all_ratings_sample", [])
    if all_ratings:
        processed_data["rating_counts"] = dict(Counter(all_ratings))

    # Process recent reviews for sentiment and selection
    recent_reviews = raw_review_data.get("recent_reviews", [])
    positive_reviews = []
    negative_reviews = []
    neutral_reviews = []

    for review in recent_reviews:
        text = review.get("review", "")
        rating = review.get("rating")
        sentiment = "neutral" # Default
        sentiment_score = 0.0

        if analyzer and text: # Check if analyzer was imported successfully
            try:
                vs = analyzer.polarity_scores(text)
                sentiment_score = vs['compound']
                if sentiment_score >= 0.05:
                    sentiment = "positive"
                    processed_data["sentiment_scores"]["positive"] += 1
                elif sentiment_score <= -0.05:
                    sentiment = "negative"
                    processed_data["sentiment_scores"]["negative"] += 1
                else:
                    sentiment = "neutral"
                    processed_data["sentiment_scores"]["neutral"] += 1
            except Exception as e:
                print(f"Sentiment analysis failed for review: {e}")
        elif rating is not None: # Fallback sentiment based on rating
             if rating >= 4:
                 sentiment = "positive"
                 processed_data["sentiment_scores"]["positive"] += 1
             elif rating <= 2:
                 sentiment = "negative"
                 processed_data["sentiment_scores"]["negative"] += 1
             else:
                 processed_data["sentiment_scores"]["neutral"] += 1


        review_details = {
            "review": text,
            "rating": rating,
            "date": review.get("date"),
            "sentiment": sentiment,
            "sentiment_score": sentiment_score # Store score for potential sorting
        }

        if sentiment == "positive":
            positive_reviews.append(review_details)
        elif sentiment == "negative":
            negative_reviews.append(review_details)
        else:
            neutral_reviews.append(review_details)

    # Select reviews: 3 positive, 2 negative (prioritize highest/lowest scores if available)
    positive_reviews.sort(key=lambda x: x.get('sentiment_score', 0), reverse=True)
    negative_reviews.sort(key=lambda x: x.get('sentiment_score', 0))

    processed_data["reviews_for_display"].extend(positive_reviews[:3])
    processed_data["reviews_for_display"].extend(negative_reviews[:2])

    # If we don't have 5 reviews yet, add neutrals or remaining ones
    needed = 5 - len(processed_data["reviews_for_display"])
    if needed > 0:
        remaining_reviews = neutral_reviews + positive_reviews[3:] + negative_reviews[2:]
        # Ensure no duplicates if a review was both neutral and in remaining pos/neg
        seen_reviews = {r['review'] for r in processed_data["reviews_for_display"]}
        for r in remaining_reviews:
            if len(processed_data["reviews_for_display"]) >= 5: break
            if r['review'] not in seen_reviews:
                processed_data["reviews_for_display"].append(r)
                seen_reviews.add(r['review'])


    print(f"Processed review data: {len(processed_data['reviews_for_display'])} reviews selected.")
    # Update state
    return {
        "processed_review_data": processed_data,
        "intent": "reviews_processed" # Signal for formatter
        }
# ---> END ADDITION <---


# ---> ADDED: ProcessCustomizationNode (from previous response) <---
def process_customization_node(state: KitchenState) -> Dict[str, Any]:
    """
    Processes the JSON output from the 'customize_recipe' tool, storing the results
    in 'customization_results' for the formatter node.
    """
    print("---NODE: ProcessCustomizationNode---")
    messages = state.get("messages", [])
    customization_tool_result = None

    # Find the last ToolMessage from customize_recipe
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "customize_recipe":
            customization_tool_result = msg
            break

    if not customization_tool_result or not customization_tool_result.content:
        print("No customization tool result found in messages.")
        return {"customization_results": None, "intent": "customization_processed_nodata"}

    try:
        # Ensure content is a string before loading JSON
        content_str = customization_tool_result.content
        if not isinstance(content_str, str):
             content_str = json.dumps(content_str)

        parsed_result = json.loads(content_str)
        print(f"Parsed customization result: {parsed_result.get('status')}")

        # Store the parsed result for the formatter
        return {
            "customization_results": parsed_result,
            "intent": "customization_processed" # Signal for formatter
        }

    except json.JSONDecodeError:
        print(f"ERROR: Could not parse customize_recipe tool result as JSON: {content_str[:100]}...")
        return {"customization_results": {"status": "error", "message": "Failed to parse customization tool output."}, "intent": "customization_error"}
    except Exception as e:
         print(f"ERROR: Unexpected error processing customization result: {e}")
         return {"customization_results": {"status": "error", "message": f"Unexpected error: {e}"}, "intent": "customization_error"}
# ---> END ADDITION <---


print("✅ LangGraph Step 5: Custom Action Nodes Defined (Revised - Nodes Restored)")


# ## 10.6. Define Conditional Edge Functions
# ### LangGraph Step 6: Directing the Flow (Conditional Edges)
# 
# Conditional edges define the branching logic within our agent's graph. They examine the agent's state after a node runs and decide which node should run next.
# 
# *   `route_after_parsing`: This is the main router after the `InputParserNode`. It checks the `intent` (set by the parser or previous nodes) or looks for tool calls and directs the flow to the `ToolExecutorNode`, one of the specific action/processing nodes (`AggregateNutritionNode`, `ReviewDashboardNode`, `ProcessCustomizationNode`), the `ResponseFormatterNode`, or `END`.
# *   `route_after_formatting`: After the `ResponseFormatterNode` prepares the text output, this edge checks if the output is a nutrition summary. If yes, it routes to the `VisualizeNutritionNode`; otherwise, it routes to `END` for this turn.
# 

# In[47]:


# LangGraph Step 6: Conditional Edge Functions (Revised)

# --- Assume KitchenState, constants are defined ---
# from step1_state import KitchenState
# from step2_core import NUTRITION_RESPONSE_HEADER

# --- Conditional Edge Functions ---

def route_after_parsing(state: KitchenState) -> Literal[
    "ToolExecutorNode", "AggregateNutritionNode",
    "ResponseFormatterNode", "ReviewDashboardNode", # ---> ADDED ReviewDashboardNode
    "ProcessCustomizationNode", # ---> ADDED ProcessCustomizationNode
    END
]:
    """
    Routes after the InputParserNode based on intent or presence of tool calls.
    - Tool calls ('tool_call' intent) -> ToolExecutorNode
    - Specific intents ('aggregate_nutrition', 'process_reviews', 'process_customization') -> Corresponding Node
    - 'exit' intent -> END
    - Otherwise (general chat, clarification, error, fetched data ready for formatter) -> ResponseFormatterNode
    """
    print("---ROUTING (After Parsing)---")
    intent = state.get("intent")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    # Check for tool calls generated by the *parser* node itself
    has_parser_tool_calls = isinstance(last_message, AIMessage) and bool(last_message.tool_calls)

    print(f"Routing based on: Intent='{intent}', HasParserToolCalls={has_parser_tool_calls}")

    if intent == "aggregate_nutrition":
        print("Routing to: AggregateNutritionNode")
        return "AggregateNutritionNode"
    # ---> ADDED Routing for review/customization processing <---
    elif intent == "process_reviews":
         print("Routing to: ReviewDashboardNode")
         return "ReviewDashboardNode"
    elif intent == "process_customization":
         print("Routing to: ProcessCustomizationNode")
         return "ProcessCustomizationNode"
    # ---> END ADDITION <---
    elif has_parser_tool_calls or intent == "tool_call": # If parser generated calls OR intent is explicitly tool_call
        print("Routing to: ToolExecutorNode")
        return "ToolExecutorNode"
    elif intent == "exit" or state.get("finished"):
        print("Routing to: END")
        return END
    else: # general_chat, clarification_needed, error, or data fetched and ready for formatter (e.g., recipe_details_fetched, live_data_fetched)
        print("Routing to: ResponseFormatterNode")
        return "ResponseFormatterNode"


# ---> REMOVED route_after_action - Replaced by more specific routing from parser/nodes <---
# The parser now directly routes ToolMessages to the appropriate processing node or back to the LLM if needed.
# Action nodes (Aggregate, ReviewDashboard, ProcessCustomization) will route directly to the Formatter.

def route_after_formatting(state: KitchenState) -> Literal["VisualizeNutritionNode", END]:
    """
    Decides whether to visualize nutrition data after formatting the response.
    Checks if the final response STARTS WITH the specific nutrition header.
    """
    print("---ROUTING (After Formatting)---")
    final_response = state.get("last_assistant_response")

    try:
        header_check = NUTRITION_RESPONSE_HEADER
    except NameError:
        print("Warning: NUTRITION_RESPONSE_HEADER not found for routing check.")
        header_check = "Here's the approximate average nutrition" # Fallback

    if final_response and final_response.strip().startswith(header_check):
        print("Routing to: VisualizeNutritionNode")
        return "VisualizeNutritionNode"
    else:
        print("Routing to: END (No visualization needed)")
        return END


print("✅ LangGraph Step 6: Conditional Edge Functions Defined (Revised)")


# ## 10.7. Assemble and Compile Graph
# ### LangGraph Step 7: Assembling the Agent (Graph Compilation & Visualization) 🕸️
# 
# It's time to put all the pieces together! We create a `StateGraph` instance and:
# 
# 1.  **Add Nodes:** Register all our defined nodes (parser, executor, action nodes, formatter, visualizer).
# 2.  **Define Edges:** Connect the nodes according to our logic, using the conditional edge functions (`route_after_parsing`, `route_after_formatting`) for branching and direct edges where the flow is fixed (e.g., `ToolExecutorNode` -> `InputParserNode`, `VisualizeNutritionNode` -> `END`).
# 3.  **Set Entry Point:** Specify that the process starts at the `InputParserNode`.
# 4.  **Compile:** Compile the graph definition into a runnable `kitchen_assistant_graph` object.
# 
# Finally, we attempt to visualize the compiled graph structure using Mermaid/Graphviz to get a clear picture of Chefbelle's internal workflow.
# 

# In[48]:


# LangGraph Step 7: Graph Assembly & Compilation (Revised)

# --- Assume KitchenState, Nodes, Edges are defined ---
# from step1_state import KitchenState (Revised)
# from step3_5_nodes import input_parser_node, response_formatter_node # (Revised)
# from step3_tools import tool_executor_node
# from step4_actions import aggregate_nutrition_node, visualize_nutrition_node, review_dashboard_node, process_customization_node # (Revised/Added)
# from step5_routing import route_after_parsing, route_after_formatting # (Revised)

# --- Graph Assembly ---
graph_builder = StateGraph(KitchenState)

# Add Nodes (Ensure names match function names)
graph_builder.add_node("InputParserNode", input_parser_node)
graph_builder.add_node("ToolExecutorNode", tool_executor_node)
graph_builder.add_node("AggregateNutritionNode", aggregate_nutrition_node)
graph_builder.add_node("ResponseFormatterNode", response_formatter_node)
graph_builder.add_node("VisualizeNutritionNode", visualize_nutrition_node)
# ---> ADDED New Nodes <---
graph_builder.add_node("ReviewDashboardNode", review_dashboard_node)
graph_builder.add_node("ProcessCustomizationNode", process_customization_node)
# ---> END ADDITION <---

# Define Entry Point
graph_builder.add_edge(START, "InputParserNode")

# Define Conditional Edges from Parser
graph_builder.add_conditional_edges(
    "InputParserNode",
    route_after_parsing,
    {
        "ToolExecutorNode": "ToolExecutorNode",
        "AggregateNutritionNode": "AggregateNutritionNode",
        "ReviewDashboardNode": "ReviewDashboardNode", # ---> ADDED <---
        "ProcessCustomizationNode": "ProcessCustomizationNode", # ---> ADDED <---
        "ResponseFormatterNode": "ResponseFormatterNode", # Handles chat, errors, fetched data
        END: END
    }
)

# ---> REVISED: Edges After Tool Execution <---
# ToolExecutorNode output (ToolMessages) goes BACK to InputParserNode
# The parser will then route based on the tool name to the appropriate processing node
# or back to the LLM if the tool result needs interpretation.
graph_builder.add_edge("ToolExecutorNode", "InputParserNode")
# ---> END REVISION <---


# ---> ADDED: Edges After Processing Nodes <---
# After aggregation, review processing, or customization processing, format the response
graph_builder.add_edge("AggregateNutritionNode", "ResponseFormatterNode")
graph_builder.add_edge("ReviewDashboardNode", "ResponseFormatterNode")
graph_builder.add_edge("ProcessCustomizationNode", "ResponseFormatterNode")
# ---> END ADDITION <---

# After formatting, decide whether to visualize or end the turn
graph_builder.add_conditional_edges(
    "ResponseFormatterNode",
    route_after_formatting,
    {
        "VisualizeNutritionNode": "VisualizeNutritionNode",
        END: END
    }
)

# After visualization, the graph run ends for this turn
graph_builder.add_edge("VisualizeNutritionNode", END)


# Compile the graph
kitchen_assistant_graph = graph_builder.compile()

print("\n✅ LangGraph Step 7: Graph Compiled Successfully! (Revised Flow)")

# Visualize the updated graph
try:
    png_data = kitchen_assistant_graph.get_graph().draw_mermaid_png()
    display(Image(png_data))
    print("Graph visualization displayed.")
except Exception as e:
    print(f"\nGraph visualization failed: {e}")


# ## 10.8. Setup UI Integration
# ### LangGraph Step 8: Setting Up the Interface (UI Simulation)
# 
# Now, let's create the `ipywidgets`-based interface to interact with our compiled LangGraph agent (`kitchen_assistant_graph`).
# 
# *   **State Management:** We initialize a `conversation_state` dictionary to hold the context between turns.
# *   **Widgets:** We create input widgets (text area, voice file dropdown), buttons, and output areas (one for the chat history, one for debug logs).
# *   **Core Logic (`run_graph_and_display`):** This function is the bridge between the UI and the agent. It takes the user's input, prepares the input state for the graph (including history), invokes the graph using `.stream()` to capture intermediate steps for debugging, updates the global `conversation_state`, and displays the user input and the agent's final response in the chat history area.
# *   **Event Handlers:** Functions (`on_text_submit`, `on_voice_submit`) are linked to the buttons to trigger the transcription (if needed) and the `run_graph_and_display` function.

# In[49]:


import ipywidgets as widgets
from IPython.display import display, Markdown, clear_output
import os
from typing import Dict, Any, List, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
# Assume other necessary imports like KitchenState, kitchen_assistant_graph, transcribe_audio are present

# LangGraph Step 8: User Interface Integration (Revised)

# --- Assume KitchenState, Graph, transcribe_audio are defined ---
# from step1_state import KitchenState (Revised)
# from step6_graph import kitchen_assistant_graph (Revised)
# from step0_utils import transcribe_audio # Or wherever it's defined

# --- UI Simulation using ipywidgets ---

# Conversation state (global for this simple example)
# Reset state for UI interaction
conversation_state: Dict[str, Any] = { # Using Dict for simplicity here, assuming KitchenState is compatible
    "messages": [], "user_input": None, "audio_file_path": None, "intent": None,
    "selected_recipe_id": None, "customization_request": None, "nutrition_query": None,
    "grounding_query": None, "current_recipe_details": None, "recipe_reviews": None,
    "ingredient_nutrition_list": None, "live_recipe_details": None, "nutritional_info": None,
    "processed_review_data": None, "customization_results": None, "grounding_results_formatted": None,
    "user_ingredients": [], "dietary_preferences": [],
    "needs_clarification": False, "finished": False, "last_assistant_response": None,
}

# Widgets
text_input = widgets.Textarea(description="You:", layout={'width': '90%', 'height': '80px'}) # Adjusted height
text_submit_button = widgets.Button(description="Send Text")

# Define voice options (use actual paths accessible to your environment)
# ---> MODIFIED: Added the two specified voice files <---
voice_options = [
    ("Select Voice...", ""),

    ("1.Intro", "/kaggle/input/voices-of-commands-genai-capstone-2025/1.Intro.mp3"),   
    ("2.Serach Recipes", "/kaggle/input/voices-of-commands-genai-capstone-2025/2.Recipe_Search.mp3"),
    ("3.Recipe Info", "/kaggle/input/voices-of-commands-genai-capstone-2025/3.Get_Recipe_Info.mp3"),
    ("4.Reviews", "/kaggle/input/voices-of-commands-genai-capstone-2025/4.Check_the_Reviews.mp3"),
    ("5.Nutrition Ino", "/kaggle/input/voices-of-commands-genai-capstone-2025/5.Get_Nutrition_Info.mp3"),
    ("6.Nutrition Analysis", "/kaggle/input/voices-of-commands-genai-capstone-2025/6.Nutrition_Analysis.mp3"),
    ("7.Recipe Customization", "/kaggle/input/voices-of-commands-genai-capstone-2025/7.Recipe_Customization.mp3"),
    ("8.Internet Searching", "/kaggle/input/voices-of-commands-genai-capstone-2025/8.Search_on_Internet.mp3"),
] 
# ---> END MODIFICATION <---

# Filter options if needed (uncomment if you want to check existence upfront)
# valid_voice_options = [(name, path) for name, path in voice_options if path is None or (path and os.path.exists(path))]
# voice_dropdown = widgets.Dropdown(options=valid_voice_options, description="Voice:")
voice_dropdown = widgets.Dropdown(options=voice_options, description="Voice:") # Using original list for now
voice_submit_button = widgets.Button(description="Process Voice")

# ---> REVISED: Output Area for Chat History <---
chat_history_output = widgets.Output(layout={
    'border': '1px solid black',
    'height': '400px',
    'overflow_y': 'scroll',
    'width': '90%'
})

# ---> REVISED: Separate Output for Debug Info <---
debug_output = widgets.Output(layout={
    'border': '1px solid blue',
    'height': '150px', # Increased height slightly
    'overflow_y': 'scroll',
    'width': '90%'
})

# Display initial welcome message in the chat history
with chat_history_output:
    display(Markdown("**Assistant:** Welcome! Ask me about recipes, ingredients, or nutrition."))
    # Optionally add to state if needed for graph context on first turn
    # conversation_state["messages"].append(AIMessage(content="Welcome! Ask me about recipes, ingredients, or nutrition."))
    # conversation_state["last_assistant_response"] = "Welcome! Ask me about recipes, ingredients, or nutrition."


# --- Interaction Logic (Revised for better display and Markdown) ---
def run_graph_and_display(initial_state_update: Dict[str, Any]):
    global conversation_state
    # Assume kitchen_assistant_graph is defined and compiled elsewhere
    global kitchen_assistant_graph

    # 1. Prepare input for the graph
    current_messages = list(conversation_state.get("messages", []))
    # The new human message is already in initial_state_update["messages"]

    input_for_graph = conversation_state.copy()
    input_for_graph.update(initial_state_update)
    # Pass the *existing* history plus the *new* human message to the graph
    input_for_graph["messages"] = current_messages + initial_state_update.get("messages", [])
    # Reset transient fields for the new run
    input_for_graph["intent"] = None
    input_for_graph["last_assistant_response"] = None
    input_for_graph["nutritional_info"] = None # Clear previous results
    input_for_graph["processed_review_data"] = None
    input_for_graph["customization_results"] = None
    input_for_graph["live_recipe_details"] = None
    input_for_graph["grounding_results_formatted"] = None
    input_for_graph["needs_clarification"] = False
    input_for_graph["finished"] = False # Reset finished flag for new input
    # Keep context like selected_recipe_id, user_ingredients, dietary_preferences

    # ---> REVISED: Display Logic - Append latest interaction <---
    user_input_text = initial_state_update.get("user_input", "")
    user_display_prefix = "**You:**"
    if initial_state_update.get("audio_file_path"):
        user_display_prefix = f"**You (from voice):**"

    with chat_history_output:
        # Display the user's input for this turn
        display(Markdown(f"{user_display_prefix} {user_input_text}"))
        # Display a thinking indicator
        thinking_output = widgets.Output()
        with thinking_output:
             display(Markdown("**Assistant:** Thinking..."))
        display(thinking_output)
    # ---> END REVISION <---

    # 2. Stream graph execution (or use invoke)
    final_state_after_run = None
    assistant_response_to_display = "..." # Default thinking message
    error_occurred = None

    # Clear previous debug info
    with debug_output:
        clear_output(wait=True)
        print("--- Running Graph ---")

    try:
        # Use stream to observe intermediate steps in debug output
        for step_state in kitchen_assistant_graph.stream(input_for_graph, {"recursion_limit": 25}):
            node_name = list(step_state.keys())[0]
            current_state_snapshot = step_state[node_name]

            # --- Debugging Output ---
            with debug_output:
                 # print(f"\n--- Step: {node_name} ---") # Keep adding steps
                 # Simple print of node name is often enough
                 print(f"-> {node_name}")
                 # Optionally print more details like intent, last message etc.
                 # intent = current_state_snapshot.get('intent')
                 # if intent: print(f"  Intent: {intent}")
                 # last_msg = current_state_snapshot.get('messages', [])[-1] if current_state_snapshot.get('messages') else None
                 # if last_msg: print(f"  Last Msg Type: {type(last_msg).__name__}")
            # --- End Debugging ---

            # Update global state progressively
            # Be careful with deep updates if KitchenState has nested dicts/lists
            # A simple update might not merge correctly. For this example, assume flat updates are okay.
            conversation_state.update(current_state_snapshot)
            final_state_after_run = conversation_state.copy() # Keep track of the latest full state

            # Check if the graph explicitly set 'finished' in this step
            if current_state_snapshot.get("finished", False):
                with debug_output: print("--- Finished flag set in step, ending stream early. ---")
                break

        # After the stream finishes (or breaks)
        if final_state_after_run:
            conversation_state.update(final_state_after_run) # Ensure final state is captured
            assistant_response_to_display = conversation_state.get("last_assistant_response", "Okay, what next?")
            # Check finished flag again after the loop completes
            if conversation_state.get("finished"):
                 assistant_response_to_display = conversation_state.get("last_assistant_response", "Goodbye!")
        else:
             # This case might happen if the stream yields nothing or input is invalid
             assistant_response_to_display = "Something went wrong during processing (no final state)."
             error_occurred = "No final state returned from graph stream."
             conversation_state["last_assistant_response"] = assistant_response_to_display
             if "messages" not in conversation_state: conversation_state["messages"] = []
             conversation_state["messages"].append(AIMessage(content=assistant_response_to_display))


    except Exception as e:
        assistant_response_to_display = f"An error occurred during graph execution: {e}"
        error_occurred = str(e)
        print(f"ERROR during graph execution: {e}")
        import traceback
        with debug_output: # Print traceback to debug area
            traceback.print_exc()
        # Update state with error message
        conversation_state["last_assistant_response"] = assistant_response_to_display
        if "messages" not in conversation_state: conversation_state["messages"] = []
        # Add error as AI message to history
        conversation_state["messages"].append(AIMessage(content=f"Error: {e}"))
        conversation_state["finished"] = True # Stop conversation on error

    # 3. Display the final AI response for this turn
    # ---> REVISED: Display Logic - Update thinking message with final response <---
    with chat_history_output:
        thinking_output.clear_output(wait=True) # Remove "Thinking..."
        if assistant_response_to_display:
            # Ensure the response is added to the state's message history if not already done by the graph
            if not conversation_state["messages"] or conversation_state["messages"][-1].content != assistant_response_to_display:
                 # Check if the last message is already the AI response to avoid duplicates
                 is_last_message_ai = isinstance(conversation_state["messages"][-1], AIMessage) if conversation_state["messages"] else False
                 if not is_last_message_ai or conversation_state["messages"][-1].content != assistant_response_to_display:
                      conversation_state["messages"].append(AIMessage(content=assistant_response_to_display))

            display(Markdown(f"**Assistant:**\n\n{assistant_response_to_display}"))
        elif error_occurred:
             display(Markdown(f"**Assistant:** Sorry, an error occurred: {error_occurred}"))
        else:
             display(Markdown(f"**Assistant:** (No response content generated)"))
        # Add a separator for clarity
        display(Markdown("---"))
    # ---> END REVISION <---

    # 4. Visualization (Matplotlib) is handled INSIDE the graph by VisualizeNutritionNode
    # and should appear in the standard cell output area below the widgets.

# --- Event Handlers (No changes needed here, they call the revised run_graph_and_display) ---
def on_text_submit(b):
    user_text = text_input.value.strip() # Use strip()
    if not user_text: return
    initial_update = {
        "user_input": user_text,
        "messages": [HumanMessage(content=user_text)], # Pass the new message
        "audio_file_path": None, # Ensure audio path is None for text input
        "finished": False
        }
    text_input.value = "" # Clear input
    run_graph_and_display(initial_update)

def on_voice_submit(b):
     selected_file = voice_dropdown.value
     if not selected_file:
         with chat_history_output: display(Markdown("**Assistant:** Please select a voice file.\n\n---")); return
     if not os.path.exists(selected_file):
          with chat_history_output: display(Markdown(f"**Assistant:** Error - Voice file not found: `{selected_file}`\n\n---")); return

     # Display processing message in chat
     with chat_history_output:
         display(Markdown(f"*(Processing voice file: {os.path.basename(selected_file)}...)*"))
         thinking_output = widgets.Output()
         with thinking_output: display(Markdown("**Assistant:** Transcribing..."))
         display(thinking_output)

     transcribed_text = "Error: Transcription setup failed."
     # --- Transcription Logic ---
     # Ensure transcribe_audio is defined and imported correctly
     # Ensure necessary API keys/credentials are set as environment variables
     try:
         # Check for transcribe_audio function existence
         if 'transcribe_audio' not in globals() or not callable(transcribe_audio):
              raise NameError("transcribe_audio function is not defined or imported.")

         google_creds = UserSecretsClient().get_secret("GOOGLE_APPLICATION_CREDENTIALS")
         # openai_key = OPENAI_API_KEY
         service_used = None
         if google_creds:
              try:
                  # from google.cloud import speech # Ensure imported if using
                  print("Attempting transcription with Google Cloud Speech...")
                  transcribed_text = transcribe_audio(service="google", file_path=selected_file, language="en",  credentials_json=google_creds)
                  service_used = "Google"
              except ImportError:
                   print("Google Cloud Speech library not installed.")
                   transcribed_text = "Error: Google Speech library missing."
              except Exception as google_err:
                   print(f"Google transcription failed: {google_err}")
                   transcribed_text = f"Error: Google transcription failed - {google_err}"
         elif openai_key:
              try:
                  # from openai import OpenAI # Ensure imported if using
                  print("Attempting transcription with OpenAI Whisper...")
                  transcribed_text = transcribe_audio(service="openai", file_path=selected_file, api_key=openai_key)
                  service_used = "OpenAI"
              except ImportError:
                   print("OpenAI library not installed.")
                   transcribed_text = "Error: OpenAI library missing."
              except Exception as openai_err:
                   print(f"OpenAI transcription failed: {openai_err}")
                   transcribed_text = f"Error: OpenAI transcription failed - {openai_err}"
         else:
              print("Neither Google Credentials nor OpenAI API Key found in environment variables.")
              transcribed_text = "Error: No transcription service configured (set GOOGLE_APPLICATION_CREDENTIALS or OPENAI_API_KEY)."

         if service_used:
             print(f"Transcription attempt finished using {service_used}.")
         else:
             print("No transcription service was successfully attempted.")

     except NameError as ne:
          print(f"ERROR: {ne}")
          transcribed_text = "Error: Transcription function missing."
     except Exception as e:
          print(f"Unexpected error during transcription setup: {e}")
          import traceback
          traceback.print_exc() # Print full traceback for unexpected errors
          transcribed_text = f"Error: Transcription failed unexpectedly - {e}"
     # --- End Transcription Logic ---


     # Clear transcribing message
     with chat_history_output:
         thinking_output.clear_output(wait=True) # Use wait=True

     if "Error:" in transcribed_text:
          with chat_history_output: display(Markdown(f"**Assistant:** Transcription failed - {transcribed_text}\n\n---"))
          return

     # Display transcribed text as user input before running graph (handled in run_graph_and_display)
     # with chat_history_output:
     #     display(Markdown(f"**You (from voice):** {transcribed_text}")) # Now handled inside run_graph

     initial_update = {
         "user_input": transcribed_text,
         "messages": [HumanMessage(content=transcribed_text)], # Pass the new message
         "audio_file_path": selected_file,
         "finished": False
     }
     voice_dropdown.value = None # Reset dropdown
     run_graph_and_display(initial_update) # Run graph with transcribed text

# Assign callbacks
text_submit_button.on_click(on_text_submit)
voice_submit_button.on_click(on_voice_submit)

# Display Widgets
print("--- Kitchen Assistant Interface ---")

# ---> REVISED: Display layout <---
# display(widgets.VBox([
#     widgets.HTML("<b>Enter request via text or select voice file:</b>"),
#     widgets.HBox([text_input, text_submit_button]), # Text input and button side-by-side
#     widgets.HBox([voice_dropdown, voice_submit_button]), # Voice input and button side-by-side
#     widgets.HTML("<hr><b>Conversation:</b>"),
#     chat_history_output, # Use the dedicated chat history output
#     widgets.HTML("<hr><b>Debug Log (Graph Steps):</b>"),
#     debug_output
# ]))

print("✅ LangGraph Step 8: UI Integration Setup Complete (Revised with added voices)")



# ## 10.9. Define Chat Helper Functions
# ### LangGraph Step 9: Chatting with Chefbelle (Helper Functions & Testing)
# 
# To test our agent more systematically outside the `ipywidgets` UI, we define helper functions:
# 
# 1.  `suppress_stdout`: A utility to temporarily hide the detailed print statements from within the nodes during testing, making the chat output cleaner.
# 2.  `chat_with_assistant`: Simulates a single turn of conversation. It takes the user message and current state, invokes the graph, suppresses the internal node prints, displays *only* the user message and the final AI response using Markdown, and returns the *updated* state for the next turn. This is great for testing conversational flow.
# 3.  `get_assistant_response_json`: Similar to `chat_with_assistant`, but designed for backend or testing scenarios where you need a structured summary of the interaction (input, response, errors, state summary) returned as a JSON string.
# 
# We'll then run a series of test scenarios using `chat_with_assistant` to validate different capabilities: greeting, recipe search, getting details, fetching reviews, nutrition analysis, customization, and handling general questions.

# In[50]:


# --- LangGraph Step 9: Chat Helper Function & JSON Output (Revised Display) ---


# Utility to suppress stdout (keep as is)
@contextlib.contextmanager
def suppress_stdout():
    # ... (keep existing code) ...
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

# --- Chat Helper with Cleaner Markdown Output ---
def chat_with_assistant(user_msg: str, state: dict) -> dict:
    """
    Runs a turn of the conversation with the LangGraph agent.
    Displays ONLY the user's input and the AI's final response for this turn,
    formatted as Markdown.
    Returns the updated state dictionary containing the full message history.
    """
    if not state:
        state = {"messages": []}
    elif "messages" not in state:
        state["messages"] = []

    # Prepare input state for the graph
    current_messages = state.get("messages", [])
    # ---> Add the human message to the history that goes INTO the graph <---
    input_messages = current_messages + [HumanMessage(content=user_msg)]

    input_for_graph = state.copy()
    input_for_graph["messages"] = input_messages
    input_for_graph["user_input"] = user_msg
    input_for_graph["finished"] = False
    # Clear fields that the graph should determine
    input_for_graph["intent"] = None
    input_for_graph["last_assistant_response"] = None
    input_for_graph["nutritional_info"] = None
    input_for_graph["processed_review_data"] = None
    input_for_graph["customization_results"] = None
    input_for_graph["live_recipe_details"] = None
    input_for_graph["needs_clarification"] = False
    # Keep context (like selected_recipe_id, current_recipe_details) from previous state

    final_state = {}
    print(f"\n--- Invoking Graph for: '{user_msg[:50]}...' ---")

    # ---> Keep suppression for cleaner notebook output during run <---
    # But ensure errors are still printed outside
    error_message = None
    with suppress_stdout():
        try:
            # Invoke the graph
            final_state = kitchen_assistant_graph.invoke(input_for_graph, {"recursion_limit": 25})
        except Exception as e:
            error_message = f"ERROR during graph invocation: {e}"
            print(error_message) # Print error outside suppressed block
            import traceback
            traceback.print_exc()
            # Create minimal error state to return
            final_state = input_for_graph.copy() # Start with input state
            error_content = f"Sorry, an error occurred: {e}"
            # Add the error message to the history
            final_state["messages"] = final_state.get("messages", []) + [AIMessage(content=error_content)]
            final_state["last_assistant_response"] = error_content
            final_state["intent"] = "error"

    print("--- Graph Invocation Complete ---")

    # --- Display ONLY the final User/AI interaction for this turn using Markdown ---

    # ---> GET the final AI response reliably from the returned state <---
    # It should have been set correctly by the revised response_formatter_node
    ai_response_text = final_state.get("last_assistant_response", None)

    # Fallback: If last_assistant_response is missing, try getting content from the last AIMessage
    if not ai_response_text and final_state.get("messages"):
         last_ai = next((msg for msg in reversed(final_state["messages"]) if isinstance(msg, AIMessage) and getattr(msg, 'content', None)), None)
         if last_ai:
             ai_response_text = last_ai.content
             print("Warning: Using content from last AIMessage as last_assistant_response was missing.") # Debug info

    # Display the user message for this turn
    display(Markdown(f"**🧑 You:**\n\n{user_msg}"))

    # Display the AI response for this turn
    if ai_response_text:
        # ---> Ensure response is treated as a string before displaying <---
        display(Markdown(f"**🤖 AI:**\n\n{str(ai_response_text)}"))
    elif error_message: # If invoke failed
         display(Markdown(f"**🤖 AI:**\n\n(Graph execution error: {error_message})"))
    else:
        display(Markdown("**🤖 AI:**\n\n(No response content generated or an error occurred)"))

    # ---> Return the final state containing the full history for the next turn <---
    # Make sure the final state has the correct message history
    # The response_formatter_node should now return a state with messages = [final_AIMessage]
    # We need to add this final AIMessage back to the history carried over from the input state
    if "messages" in final_state and isinstance(final_state["messages"], list) and len(final_state["messages"]) == 1 and isinstance(final_state["messages"][0], AIMessage):
         # Correct: formatter returned only the last AI message. Reconstruct history.
         final_state["messages"] = input_messages + final_state["messages"]
    elif error_message and "messages" in final_state:
        # Correct: error message was already added to input_messages history
        pass
    else:
         # Fallback/Warning: If message history is not as expected, log it
         #print(f"Warning: Message history structure from graph might be unexpected. Type: {type(final_state.get('messages'))}, Length: {len(final_state.get('messages', []))}")
         # Attempt to preserve input history + add the text response if possible
         if ai_response_text and not error_message:
             final_state["messages"] = input_messages + [AIMessage(content=str(ai_response_text))]
         else: # Preserve input history at least
             final_state["messages"] = input_messages


    return final_state

# --- JSON Output Function (Add similar safety checks) ---
def get_assistant_response_json(user_msg: str, state: dict) -> str:
    """
    Runs a turn of the conversation and returns the key information
    (user input, AI response, final state summary) as a JSON string.
    """
    # ... (Input state preparation - same as chat_with_assistant) ...
    if not state: state = {"messages": []}
    elif "messages" not in state: state["messages"] = []
    current_messages = state.get("messages", [])
    input_messages = current_messages + [HumanMessage(content=user_msg)]
    input_for_graph = state.copy(); input_for_graph["messages"] = input_messages
    input_for_graph["user_input"] = user_msg; input_for_graph["finished"] = False
    input_for_graph["intent"] = None; input_for_graph["last_assistant_response"] = None # etc.

    final_state = {}
    error_message = None
    with suppress_stdout(): # Suppress internal prints
        try:
            final_state = kitchen_assistant_graph.invoke(input_for_graph, {"recursion_limit": 50})
        except Exception as e:
            error_message = str(e)
            final_state = input_for_graph.copy()
            final_state["intent"] = "error"
            final_state["last_assistant_response"] = f"Error: {e}"
            # Add error to message history for completeness if needed for JSON output
            final_state["messages"] = final_state.get("messages", []) + [AIMessage(content=f"Error: {e}")]


    # Extract relevant info
    ai_response_text = final_state.get("last_assistant_response", None)
    if not ai_response_text and not error_message and final_state.get("messages"):
         last_ai = next((msg for msg in reversed(final_state["messages"]) if isinstance(msg, AIMessage) and getattr(msg, 'content', None)), None)
         if last_ai: ai_response_text = last_ai.content

    # ---> Ensure state_summary accesses fields safely using .get() <---
    state_summary = {
        "intent": final_state.get("intent"),
        "selected_recipe_id": final_state.get("selected_recipe_id"),
        "needs_clarification": final_state.get("needs_clarification", False),
        "finished": final_state.get("finished", False),
        "has_recipe_details": bool(final_state.get("current_recipe_details")),
        "has_live_details": bool(final_state.get("live_recipe_details")),
        "has_reviews": bool(final_state.get("recipe_reviews")), # Raw reviews might still be present
        "has_processed_reviews": bool(final_state.get("processed_review_data")),
        "has_nutrition": bool(final_state.get("nutritional_info")),
        "has_customization": bool(final_state.get("customization_results")),
    }

    # Construct final output JSON data
    output_data = {
        "user_input": user_msg,
        "ai_response": str(ai_response_text) if not error_message and ai_response_text else None,
        "error": error_message,
        "state_summary": state_summary,
        # Optionally include message history, converting BaseMessage objects
        # "message_history": [msg.dict() for msg in final_state.get("messages", [])] # Be careful with serialization
    }

    try:
        # Use default=str for better serialization safety
        return json.dumps(output_data, indent=2, default=str)
    except TypeError as e:
        return json.dumps({"error": "Failed to serialize final state to JSON.", "details": str(e)}, indent=2)


print("✅ LangGraph Step 9: Chat Helper Function & JSON Output Defined (Revised Display)")


# ## 10.9.1. Initializing the Conversation
# 
# Let's start a fresh conversation with Chefbelle by initializing an empty state dictionary.

# In[51]:


state = {}  # Initialize empty state first


# ## 10.9.2. Test Scenario 1: The First "Hello" 👋
# 
# Let's see how Chefbelle responds to an initial greeting. According to our system prompt, she should introduce herself briefly.
# Runs the first turn of the conversation using the `chat_with_assistant` helper. Sends a greeting ("Hello, what can you do?") to the agent and displays the interaction. Updates the `state` variable.
# 
# 
# 

# In[52]:


state = chat_with_assistant("Hello, what can you do?", state)


# In[ ]:





# ## 10.9.3. Test Scenario 2: Recipe Discovery - Vegetarian Soup Quest 🍲
# 
# Now, let's ask Chefbelle to find some recipes. We'll request vegetarian soup recipes and ask for 5 results. This should trigger the `gemini_recipe_similarity_search` tool.
# .

# In[53]:


state = chat_with_assistant("Find me vegeterian soup recipes and tag the recipes. 5 recipes", state)


# ## 10.9.4. Test Scenario 3: Getting the Details - Diving Deeper 📖
# 
# Chefbelle provided a list. Let's ask for details about the third recipe mentioned. She should use the context (`selected_recipe_id`) and call the `get_recipe_by_id` tool, displaying the Recipe Dashboard.

# In[54]:


state = chat_with_assistant("I like to know about third recipe in the list but not review", state)


# In[ ]:





# ## 10.9.5. Test Scenario 4: Hearing from Others - Fetching Reviews ⭐
# 
# How did other cooks find this recipe? Let's ask for reviews. This should trigger `get_ratings_and_reviews_by_recipe_id`, followed by the `ReviewDashboardNode` and the formatted Review Dashboard output.

# 

# In[55]:


state = chat_with_assistant("Show the recipe reviews", state)


# In[ ]:





# ## 10.9.6. Test Scenario 5: Nutritional Breakdown - Health Insights 📊
# 
# Time for the nutrition analysis! Asking for nutrition information should trigger multiple calls to `fetch_nutrition_from_usda_fdc` (one per ingredient), followed by `AggregateNutritionNode`, `ResponseFormatterNode` (for the summary), and finally `VisualizeNutritionNode` (for the plot).

# In[56]:


state = chat_with_assistant("Get nutriotion information for this recipe", state)


# In[57]:


state = chat_with_assistant("Run the nutrition analysis for the recipe we just discussed.", state)


# In[58]:


state = chat_with_assistant("Get nutriotion information for this recipe", state)


# In[59]:


# state = chat_with_assistant("yes it is that same recipe", state)


# ## 10.9.7. Test Scenario 6: Making it Your Own - Recipe Customization 🔧
# 
# Let's ask Chefbelle to adapt the current recipe. We'll request a healthier, low-fat version. This should trigger the `customize_recipe` tool and the `ProcessCustomizationNode`.

# In[60]:


state = chat_with_assistant("make this recipe more healthy for low fat diet", state)


# In[ ]:





# ## 10.9.8. Test Scenario 7: General Knowledge - Grounding Query 🤔
# 
# What if we ask something outside the recipe database? Let's inquire about egg yolk substitutes. Chefbelle should use her internal knowledge, potentially augmented by the built-in Google Search grounding.

# In[61]:


text = "What's a good substitute for egg yolks"
state = chat_with_assistant(text, state)


# In[ ]:





# ## 10.9.9 Test Scenario 8: Grounding Query By Voice Command 
# 
# This cell demonstrates processing a single, pre-recorded voice command:
# 
# 1.  **Specify Audio File:**
#     *   The `voice_path` variable is set to the location of a specific MP3 audio file (`/kaggle/input/.../8.Search_on_Internet.mp3`) within the Kaggle input directory. This file presumably contains a voice command related to searching the internet.
# 
# 2.  **Transcribe Audio:**
#     *   The `transcribe_audio()` function is called to convert the audio file at `voice_path` into text.
#     *   `service="google"` selects Google Cloud Speech-to-Text for transcription.
#     *   `credentials_json=SecretValueJson` provides the necessary Google Cloud credentials (assumed to be loaded from Kaggle Secrets into the `SecretValueJson` variable).
#     *   The resulting text transcription is stored in the `voice_command` variable.
# 
# 3.  **Interact with Assistant:**
#     *   The `chat_with_assistant()` function is called, passing the transcribed `voice_command` as input.
#     *   It also takes the current conversation `state` (which might include history or context).
#     *   The function processes the command and returns the updated conversation `state`, which overwrites the previous `state`.
# 
# **Purpose:** To test the transcription and assistant interaction pipeline using a specific voice input file.
# 
# **Assumptions:**
# *   The `transcribe_audio` and `chat_with_assistant` functions are defined elsewhere in the notebook.
# *   The `SecretValueJson` variable holds valid Google Cloud credentials JSON string from Kaggle Secrets.
# *   The `state` variable has been initialized.
# *   The specified audio file exists at the `voice_path`.

# In[62]:


voice_path = "/kaggle/input/voices-of-commands-genai-capstone-2025/8.Search_on_Internet.mp3"
voice_command = transcribe_audio(service="google", file_path=voice_path, credentials_json=SecretValueJson)
state = chat_with_assistant(voice_command, state)


# Chat with Voice

# In[ ]:





# In[ ]:





# ### 10.9.10 The Full Conversation: Final Message History
# 
# Let's examine the complete message history stored in the final state after our test conversation. This shows the flow of user requests and AI responses, including tool calls represented internally.

# 

# In[63]:


state["messages"]


# ### LangGraph Step 11 (Optional): Testing Voice Input with JSON Output
# 
# This section defines a helper function `chat_with_voice_and_get_json` specifically for testing the voice input flow and getting a structured JSON summary of the interaction turn. This is useful for automated testing or backend integration where you might not need the interactive chat display.
# 
# *It transcribes the audio, invokes the graph using `get_assistant_response_json` (which suppresses node prints), and returns a JSON object containing the transcribed input, AI response, any errors, and a summary of the agent's state.*
# 
# *Note: This function uses the provided state for context but does **not** update the main `state` variable in the notebook after its run.*

# In[64]:


# LangGraph Step 10: Testing Voice Input with JSON Output



# --- Assume transcribe_audio and get_assistant_response_json are defined ---
# Make sure these functions are defined in previous cells or imported

# --- Voice-to-JSON Helper Function ---
def chat_with_voice_and_get_json(
    audio_file_path: str,
    state: dict,
    service: str = "google", # Default to google as used in your test
    language: str = "en",
    api_key: Optional[str] = None, # For OpenAI
    credentials_path: Optional[str] = None, # For Google Path
    credentials_json: Optional[str] = None # For Google JSON String (if used)
) -> str:
    """
    Transcribes an audio file, runs the transcription through the agent,
    and returns key interaction details as a JSON string.
    Uses the provided 'state' for context during the graph run for this turn,
    but does NOT return the updated state dictionary.
    """
    print(f"\n--- Starting Voice-to-JSON process for: {audio_file_path} ---")

    # 1. Validate Audio File Path
    if not audio_file_path:
        print("ERROR: Audio file path is required.")
        return json.dumps({"error": "Audio file path is required."}, indent=2)
    if not os.path.exists(audio_file_path):
        print(f"ERROR: Audio file not found at: {audio_file_path}")
        return json.dumps({"error": f"Audio file not found at: {audio_file_path}"}, indent=2)

    # 2. Transcribe Audio
    print(f"--- Transcribing audio file using {service}... ---")
    transcribed_text = transcribe_audio(
        service=service,
        file_path=audio_file_path,
        language=language,
        api_key=api_key,
        credentials_path=credentials_path,
        credentials_json=credentials_json
    )
    print(f"--- Transcription complete ---")

    # 3. Handle Transcription Errors
    if not transcribed_text or transcribed_text.startswith("Error:"):
        print(f"Transcription Error/Empty: {transcribed_text}")
        return json.dumps({
            "user_input_source": "voice",
            "audio_file": audio_file_path,
            "transcription_error": transcribed_text,
            "ai_response": None,
            "error": "Transcription failed",
            "state_summary": None
            }, indent=2)

    print(f"Transcribed Text: {transcribed_text}")

    # 4. Invoke Graph via get_assistant_response_json
    # This function internally invokes the graph using the provided 'state' for context.
    print(f"--- Invoking Graph with transcribed text ('{transcribed_text[:50]}...') for JSON output ---")
    json_output = get_assistant_response_json(user_msg=transcribed_text, state=state)
    print(f"--- Graph invocation for JSON complete ---")

    # 5. Add transcribed text to the output JSON and return
    try:
        output_data = json.loads(json_output)
        output_data["transcribed_input"] = transcribed_text # Add the transcription
        return json.dumps(output_data, indent=2, default=str)
    except json.JSONDecodeError:
        print("Error: Could not parse the JSON output from get_assistant_response_json")
        return json.dumps({
            "error": "Failed to parse internal JSON response",
            "raw_output": json_output,
            "transcribed_input": transcribed_text
            }, indent=2)
    except Exception as e:
         print(f"Unexpected error modifying JSON output: {e}")
         return json.dumps({
            "error": f"Unexpected error processing JSON: {e}",
            "raw_output": json_output,
            "transcribed_input": transcribed_text
            }, indent=2)

print("✅ LangGraph Step 11: Voice-to-JSON Helper Function Defined")

# --- Test Scenario: Voice Input to JSON Output ---

# Initialize or use the existing state from previous text interactions
# If you ran the text scenarios in Step 9, 'state' will contain that history.
# state = {} # Uncomment to start with a fresh, empty state

# Define the path to your audio file
# Make sure this path is correct for your environment
audio_file = "/home/snowholt/coding/python/google_capstone/voices/Nariman_1.ogg"

# Define credentials/keys (ensure these are correctly set in your environment/secrets)


# Choose the service ('google' or 'openai')
transcription_service = "google" # Or "openai"

# Call the new function
# if transcription_service == "google":
#     if not google_creds_path or not os.path.exists(google_creds_path):
#          print("ERROR: GOOGLE_APPLICATION_CREDENTIALS path not set or invalid.")
#          json_result = json.dumps({"error": "Google credentials path missing or invalid."}, indent=2)
#     else:
#         json_result = chat_with_voice_and_get_json(
#             audio_file_path=audio_file,
#             state=state, # Pass the current state for context
#             service="google",
#             language="en-US", # Use appropriate code for Google
#             credentials_path=google_creds_path
#         )
# elif transcription_service == "openai":
#      if not openai_api_key:
#          print("ERROR: OPENAI_API_KEY not set.")
#          json_result = json.dumps({"error": "OpenAI API key missing."}, indent=2)
#      else:
#          json_result = chat_with_voice_and_get_json(
#             audio_file_path=audio_file,
#             state=state, # Pass the current state for context
#             service="openai",
#             language="en", # Use appropriate code for OpenAI
#             api_key=openai_api_key
#         )
# else:
#     json_result = json.dumps({"error": "Invalid transcription service selected"}, indent=2)


# # Print the resulting JSON
# print("\n--- JSON Output from Voice Interaction ---")
# print(json_result)

# # --- Note on State Update ---
# # The 'state' variable in this notebook cell *has not* been updated by the
# # chat_with_voice_and_get_json call because it returns a JSON string, not the state dict.
# # To continue the conversation programmatically after this voice turn, you would need to:
# # 1. Modify chat_with_voice_and_get_json to return the full state dict.
# # OR
# # 2. Manually parse 'json_result' and update the 'state' dictionary with the new messages.
# # OR
# # 3. Use the alternative approach: transcribe first, then call chat_with_assistant.
# print("\n--- Note: Notebook 'state' variable is not updated by this function call. ---")


# In[ ]:





# In[65]:


# To save as MP3, you might need a specific backend and additional libraries
# and the process can be more involved. WAV is generally better supported.


# ## Phase 12: Plating the Dish - Conclusion & Future Directions 🍽️✨
# 
# We've reached the final phase of our Chefbelle development journey documented in this notebook! From gathering the raw "ingredients" (data) to building Chefbelle's "brain" (the LangGraph agent) and giving her "ears" (audio input), we've laid a robust foundation for an intelligent and interactive AI kitchen assistant.
# 
# ### 12.1. The Journey Recap 🗺️
# 
# This notebook has walked through the essential steps:
# 
# 1.  📊 **Data Foundation:** Acquiring, exploring, cleaning, and preprocessing recipe, interaction, and nutrition datasets.
# 2.  🧠 **Memory Construction:** Setting up both a structured SQLite database for precise lookups and a ChromaDB vector database for powerful semantic search.
# 3.  🛠️ **Tooling Up:** Defining a suite of tools for Chefbelle to interact with databases, external APIs (like USDA Nutrition), and perform custom logic (like placeholders for customization and live data fetching).
# 4.  🎤 **Sensory Input:** Integrating audio transcription capabilities to understand voice commands.
# 5.  🤖 **Agent Architecture:** Designing and compiling a stateful LangGraph agent capable of managing conversation context, routing tasks, calling tools, and processing results.
# 6.  🖥️ **Interaction Simulation:** Building a basic `ipywidgets` interface to test the end-to-end user interaction flow.
# 
# ### 12.2. Key Accomplishments 🏆
# 
# Through this process, we've successfully built a prototype demonstrating Chefbelle's core potential:
# 
# *   🍳 **Ingredient-Based Discovery:** Leveraging semantic search via ChromaDB to find recipes based on user queries.
# *   🥗 **Dietary Awareness:** Implementing basic dietary tagging and filtering capabilities.
# *   📊 **Nutritional Insights:** Integrating with the USDA API via a dedicated tool to fetch reliable ingredient nutrition data and providing aggregated summaries.
# *   🗣️ **Natural Interaction:** Enabling both text and voice input processing.
# *   🧠 **Intelligent Orchestration:** Using LangGraph to manage complex conversational flows involving multiple tool calls and state management.
# *   🌐 **Grounded Knowledge:** Utilizing Gemini's built-in Google Search grounding for general cooking questions.
# 
# ### 12.3. Current Limitations & Challenges 🚧
# 
# This notebook represents a significant step, but it's important to acknowledge areas for improvement:
# 
# *   **Placeholder Tools:** The `customize_recipe` and `fetch_live_recipe_data` tools are currently placeholders and require more sophisticated LLM logic or web scraping implementations.
# *   **Basic Tagging:** Dietary and cuisine tagging relies on simple keyword matching and could be enhanced with more advanced NLP or ML models.
# *   **Simulated UI:** The `ipywidgets` interface is for demonstration; a full web or mobile application is needed for a real-world user experience.
# *   **Static Context:** User preferences and pantry inventory are not yet persistently stored or dynamically updated.
# *   **Scalability:** Handling larger datasets or more complex user interactions in the vector DB and agent logic might require optimization.
# 
# ### 12.4. The Next Course: Future Work 🚀
# 
# Chefbelle is just getting started! We have ambitious plans to enhance her capabilities:
# 
# *   🧺 **Pantry Integration:** Connect Chefbelle to a user's pantry inventory (manual input or smart fridge integration) to track available ingredients accurately.
# *   📸 **Multi-modal Input:** Allow users to upload images of ingredients for identification.
# *   📅 **Proactive Suggestions:** Enable Chefbelle to suggest recipes based on expiring items in the pantry.
# *   🎯 **Advanced Personalization:** Develop robust user profiles to store dietary needs, allergies, taste preferences, and cooking skill levels for truly tailored recommendations.
# *   📈 **Interactive Dashboards:** Create dynamic dashboards within a web/mobile app for meal planning, nutrition tracking over time, and shopping list generation.
# *   🕸️ **Expanded Knowledge:** Implement robust web scraping (respecting `robots.txt`) from food.com and other recipe sites to continuously expand and update the recipe database.
# *   🔄 **Database Updates:** Establish pipelines for regularly updating the vectorized and SQL databases with new recipes and nutritional information.
# *   🛒 **Ingredient Procurement:** Add options for users to directly purchase missing ingredients online via grocery delivery APIs.
# *   🧑‍🌾 **Local Sourcing:** Integrate referrals or information about local markets for fresh ingredient shopping.
# *   ⚙️ **Real-Time Backend:** Migrate the agent logic to a scalable backend using LangServe or similar frameworks for real-time, persistent agent responses.
# *   📱 **Mobile & Accessibility:** Develop dedicated mobile applications (iOS/Android) and focus on accessibility features (e.g., enhanced voice control, screen reader compatibility).
# 
# ### 12.5. Bon Appétit! 🎉
# 
# This notebook documents the creation of Chefbelle – a proof-of-concept for an AI kitchen assistant designed to make cooking more intuitive, personalized, and less wasteful. While there's more cooking to do, the foundation is set for a truly helpful culinary companion. We're excited about the future possibilities and the potential for Chefbelle to transform the everyday kitchen experience.
# 
# ---
# *Thank you for following along on this development journey!*
