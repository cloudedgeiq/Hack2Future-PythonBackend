import ollama
import base64
import os

# --- Configuration ---
OLLAMA_SERVER_IP = "10.0.0.7"
OLLAMA_SERVER_PORT = 11434
OLLAMA_MODEL_NAME = "llava"  # Or "llava:latest", "llava:7b", etc.
IMAGE_FILE_PATH = "Image1.jpeg" # IMPORTANT: Change this to the actual path of your image

# --- Helper Function ---
def image_to_base64(filepath):
    """Converts an image file to a base64 encoded string."""
    try:
        with open(filepath, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def main():
    print(f"Attempting to connect to Ollama server at http://{OLLAMA_SERVER_IP}:{OLLAMA_SERVER_PORT}")

    # 1. Initialize the Ollama client
    try:
        client = ollama.Client(host=f"http://{OLLAMA_SERVER_IP}:{OLLAMA_SERVER_PORT}")
        # Quick test to see if client can list models (checks connectivity and server response)
        client.list()
        print(f"Successfully connected to Ollama server.")
    except Exception as e:
        print(f"Error connecting to Ollama server: {e}")
        print("Please ensure:")
        print(f"  1. Ollama is running on {OLLAMA_SERVER_IP}:{OLLAMA_SERVER_PORT}.")
        print(f"  2. The OLLAMA_HOST environment variable on the server is set to '0.0.0.0'.")
        print(f"  3. The firewall on {OLLAMA_SERVER_IP} allows incoming connections on port {OLLAMA_SERVER_PORT}.")
        return

    # 2. Load and encode the image
    if not os.path.exists(IMAGE_FILE_PATH):
        print(f"Error: Image file '{IMAGE_FILE_PATH}' not found on this VM (10.0.0.6).")
        print("Please place the image in the same directory as this script or provide the full path.")
        return

    print(f"Loading image from: {IMAGE_FILE_PATH}")
    base64_image = image_to_base64(IMAGE_FILE_PATH)

    if not base64_image:
        return

    # 3. Prepare the prompt for LLaVA
    # LLaVA is a multimodal model, so you "chat" with it, providing the image.
    prompt_text = "Transcribe the handwritten text in this image. Provide only the text."

    print(f"Sending request to LLaVA model ('{OLLAMA_MODEL_NAME}') for OCR...")

    try:
        response = client.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {
                    'role': 'user',
                    'content': prompt_text,
                    'images': [base64_image]  # Pass the base64 encoded image
                }
            ]
        )

        transcribed_text = response['message']['content']

        print("\n--- LLaVA OCR Result ---")
        print(transcribed_text)
        print("------------------------")

    except ollama.ResponseError as e:
        print(f"Ollama API Error: {e.status_code} - {e.error}")
        if "model not found" in str(e.error).lower():
            print(f"Hint: Make sure the model '{OLLAMA_MODEL_NAME}' is pulled on the Ollama server ({OLLAMA_SERVER_IP}).")
            print(f"You can pull it using: ollama pull {OLLAMA_MODEL_NAME}")
    except Exception as e:
        print(f"An unexpected error occurred during the Ollama request: {e}")

if __name__ == "__main__":
    # --- !!! IMPORTANT: UPDATE THIS PATH !!! ---
    # Place your handwritten image in the same directory as this script
    # and name it 'handwritten_image.png', or update the path below.
    # Example: IMAGE_FILE_PATH = "C:/Users/YourUser/Desktop/my_notes.jpg"
    # Example: IMAGE_FILE_PATH = "images/scan001.png"

    IMAGE_FILE_PATH = "Image1.jpeg" # Default if in same directory
    # -------------------------------------------
    main()