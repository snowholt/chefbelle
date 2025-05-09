#!/bin/bash

# Setup script for Chefbelle project
# This script helps new contributors get their environment ready

echo "🔧 Setting up Chefbelle development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else 
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env file for API keys if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔑 Creating .env file for API keys..."
    cat > .env << EOL
# Add your API keys below
GOOGLE_API_KEY=your_gemini_key_here
USDA_API_KEY=your_usda_key_here
# Uncomment if using OpenAI Whisper
# OPENAI_API_KEY=your_openai_key_here
EOL
    echo "⚠️ Please edit the .env file with your actual API keys"
else
    echo "✅ .env file already exists"
fi

echo "✨ Setup complete! You're ready to start contributing to Chefbelle."
echo "To activate the environment in the future, run: source venv/bin/activate"
