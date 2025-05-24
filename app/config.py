# app/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
# In production, you would likely configure these directly in the environment
load_dotenv()

# --- Flask Configuration ---
SECRET_KEY = os.getenv('SECRET_KEY', 'a_default_secret_key_if_not_set') # Change in production!
DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
HOST = os.getenv('HOST', '127.0.0.1')
PORT = int(os.getenv('PORT', 5000))

# --- Azure OpenAI Configuration ---
# Ensure these match the variable names used in your .env file and scripts
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview") # Use a version supporting vision
# These should match your deployment names in Azure OpenAI Studio
GPT4O_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME")
GPT4O_MINI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_GPT4O_MINI_DEPLOYMENT_NAME") # Add if you use 4o-mini deployment
O1_MINI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_O1_MINI_DEPLOYMENT_NAME") # Add if you use o1-mini deployment

# --- Azure Blob Storage Configuration (Placeholder) ---
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

# --- Tool Specific Configurations ---
# For diagram analysis: path to the reference map
# Consider making this configurable or fetching from storage
REFERENCE_MAP_PATH = os.getenv("REFERENCE_MAP_PATH", "path/to/your/correct_map.jpg")
# Ensure this reference map exists or is handled!

# --- Validation ---
# Simple check for required Azure keys for the tools
AZURE_READY = all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, GPT4O_DEPLOYMENT_NAME])
if not AZURE_READY:
    print("Warning: Azure OpenAI credentials or deployment name not fully configured.")
    print("Some features may not work.")
    print(f"  AZURE_OPENAI_API_KEY: {'Set' if AZURE_OPENAI_API_KEY else 'NOT SET'}")
    print(f"  AZURE_OPENAI_ENDPOINT: {'Set' if AZURE_OPENAI_ENDPOINT else 'NOT SET'}")
    print(f"  GPT4O_DEPLOYMENT_NAME: {'Set' if GPT4O_DEPLOYMENT_NAME else 'NOT SET'}")
    # exit(1) # Don't exit immediately in Flask, just log warning and disable feature