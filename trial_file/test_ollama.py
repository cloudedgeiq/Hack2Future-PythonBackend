# C:\Temp\test_ollama.py
import ollama
import sys

print(f"Python executable: {sys.executable}")
print(f"Ollama library path: {ollama.__file__}")
# print(f"Ollama library version: {ollama.__version__}") # If version attribute exists

try:
    client = ollama.Client(host="http://10.0.0.7:11434") # Use a dummy host for now
    print("Successfully created ollama.Client object.")
    print(client)
except Exception as e:
    print(f"Error creating ollama.Client: {e}")

# Test if ResponseError exists
try:
    print(ollama.ResponseError)
    print("ollama.ResponseError attribute exists.")
except AttributeError:
    print("ollama.ResponseError attribute DOES NOT exist.")