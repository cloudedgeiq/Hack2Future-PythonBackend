import os
import base64
from openai import AzureOpenAI
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# This is your DEPLOYMENT NAME for the GPT-4o model in Azure OpenAI Studio
GPT4O_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME")

# Ensure you are using an API version that supports GPT-4o with vision.
# Check the Azure OpenAI documentation for the latest recommended version.
# As of mid-2024, "2024-02-01" or "2024-05-01-preview" (or newer) should work.
AZURE_OPENAI_API_VERSION = "2024-05-01-preview" # Or your preferred version

# --- Helper Functions ---
def encode_image_to_base64(image_path):
    """Encodes a local image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def get_image_mime_type(image_path):
    """Determines the MIME type of an image based on its extension."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        return "image/png"
    elif ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".gif":
        return "image/gif"
    elif ext == ".webp":
        return "image/webp"
    else:
        # Default or raise an error if you want to be strict
        print(f"Warning: Unknown image type for extension {ext}. Defaulting to image/jpeg.")
        print("Supported formats typically include PNG, JPEG, GIF, WEBP.")
        return "image/jpeg"

# --- Main OCR Function ---
def ocr_with_azure_gpt4o(image_path_or_url, prompt="Extract all text from this image."):
    """
    Performs OCR on an image using Azure OpenAI GPT-4o.

    Args:
        image_path_or_url (str): Path to a local image file or a public URL of an image.
        prompt (str): The instruction for the model.

    Returns:
        str: The extracted text or an error message.
    """
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, GPT4O_DEPLOYMENT_NAME]):
        return "Error: Azure OpenAI credentials or deployment name not configured. Please check your .env file."

    try:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

        image_data_url = ""
        if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
            # If it's a URL, GPT-4o can fetch it directly
            image_data_url = image_path_or_url
            print(f"Using image URL: {image_data_url}")
        else:
            # If it's a local path, encode it
            print(f"Using local image path: {image_path_or_url}")
            base64_image = encode_image_to_base64(image_path_or_url)
            if not base64_image:
                return "Error: Could not encode local image."
            mime_type = get_image_mime_type(image_path_or_url)
            image_data_url = f"data:{mime_type};base64,{base64_image}"

        print("Sending request to Azure OpenAI GPT-4o...")
        response = client.chat.completions.create(
            model=GPT4O_DEPLOYMENT_NAME,  # Your GPT-4o deployment name
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url,
                                # "detail": "low" # or "high" or "auto" - "high" might be better for OCR
                            },
                        },
                    ],
                }
            ],
            max_tokens=2000  # Adjust as needed based on expected text length
        )
        print("Received response.")
        return response.choices[0].message.content

    except Exception as e:
        return f"An API error occurred: {e}"

# --- Example Usage ---
if __name__ == "__main__":
    # --- OPTION 1: Local Image File ---
    # Replace with the actual path to your image
    # local_image_path = "Chemistry-balanceequaltion.jpg" # <--- !!! REPLACE WITH YOUR IMAGE PATH !!!
    # local_image_path = "BadImage.jpg" # <--- !!! REPLACE WITH YOUR IMAGE PATH !!!
    local_image_path = "math.png" # <--- !!! REPLACE WITH YOUR IMAGE PATH !!!
    # local_image_path = "Chemistry-balanceequaltion.jpg" # <--- !!! REPLACE WITH YOUR IMAGE PATH !!!

    # Create a dummy image for testing if the specified one doesn't exist
    # and Pillow is installed
    if not os.path.exists(local_image_path) and not (local_image_path.startswith("http://") or local_image_path.startswith("https://")):
        try:
            from PIL import Image, ImageDraw, ImageFont
            print(f"'{local_image_path}' not found. Attempting to create a dummy image for testing...")
            img = Image.new('RGB', (600, 150), color = (255, 255, 255))
            d = ImageDraw.Draw(img)
            try:
                # Try to load a common font, fallback to default
                font = ImageFont.truetype("arial.ttf", 40)
            except IOError:
                font = ImageFont.load_default()
            d.text((10,10), "Hello Azure GPT-4o!\nThis is a test OCR image.", fill=(0,0,0), font=font)
            
            # Ensure directory exists if path includes directories
            os.makedirs(os.path.dirname(local_image_path) or '.', exist_ok=True)
            img.save(local_image_path)
            print(f"Dummy image saved as '{local_image_path}'")
        except ImportError:
            print("Pillow library not installed. Cannot create a dummy image.")
            print(f"Please provide a valid path for 'local_image_path' or install Pillow: pip install Pillow")
        except Exception as e:
            print(f"Could not create dummy image: {e}")

    if os.path.exists(local_image_path) or local_image_path.startswith("http://") or local_image_path.startswith("https://"):
        print(f"\n--- OCR for: {local_image_path} ---")
        extracted_text_local = ocr_with_azure_gpt4o(local_image_path)
        print("\nExtracted Text:")
        print(extracted_text_local)
    else:
        print(f"\nSkipping local image OCR as '{local_image_path}' was not found and could not be created.")


    # --- OPTION 2: Image URL ---
    # Example public image URL (e.g., a sign or a document scan)
    # image_url = "https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png"
    # print(f"\n--- OCR for URL: {image_url} ---")
    # extracted_text_url = ocr_with_azure_gpt4o(image_url, prompt="What text is on this presentation slide?")
    # print("\nExtracted Text from URL:")
    # print(extracted_text_url)

    # --- OPTION 3: More specific prompt example ---
    # if os.path.exists(local_image_path) or local_image_path.startswith("http://") or local_image_path.startswith("https://"):
    #     print(f"\n--- OCR for {local_image_path} (specific prompt) ---")
    #     specific_prompt = "This is an invoice. Extract the invoice number, date, and total amount."
    #     extracted_data = ocr_with_azure_gpt4o(local_image_path, prompt=specific_prompt)
    #     print("\nExtracted Data (with specific prompt):")
    #     print(extracted_data)