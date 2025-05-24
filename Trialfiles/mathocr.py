import os
import base64
import argparse
from openai import AzureOpenAI
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont # For dummy image creation

# --- Configuration ---
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
GPT4O_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = "2024-05-01-preview" # Or your preferred, vision-compatible version

# --- Helper Functions ---
def encode_image_to_base64(image_path):
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
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png": return "image/png"
    elif ext in [".jpg", ".jpeg"]: return "image/jpeg"
    # Add other types if needed
    else:
        print(f"Warning: Unknown image type for {ext}. Defaulting to image/jpeg.")
        return "image/jpeg"

def initialize_azure_client():
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, GPT4O_DEPLOYMENT_NAME]):
        print("Error: Azure OpenAI credentials or deployment name not configured. Please check your .env file.")
        return None
    try:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        return client
    except Exception as e:
        print(f"Error initializing Azure OpenAI client: {e}")
        return None

# --- Stage 1: Initial OCR ---
def perform_initial_ocr(client, image_path_or_url, detail="high"):
    """Performs initial OCR on an image using Azure OpenAI GPT-4o."""
    print(f"\n--- STAGE 1: Performing Initial OCR on: {image_path_or_url} ---")

    image_data_url = ""
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        image_data_url = image_path_or_url
    else:
        base64_image = encode_image_to_base64(image_path_or_url)
        if not base64_image:
            return "Error: Could not encode local image for OCR."
        mime_type = get_image_mime_type(image_path_or_url)
        image_data_url = f"data:{mime_type};base64,{base64_image}"

    ocr_prompt = "Carefully extract all handwritten text and mathematical expressions from this image. Preserve the structure and symbols as accurately as possible, even if they seem like fragments. Pay close attention to all symbols, including plus, minus, equals, square roots, exponents, and fractions."

    try:
        response = client.chat.completions.create(
            model=GPT4O_DEPLOYMENT_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": ocr_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url, "detail": detail}}
                ],
            }],
            max_tokens=2000, # Adjust as needed
            temperature=0.1  # Low temperature for literal transcription
        )
        ocr_result = response.choices[0].message.content
        print("Initial OCR completed.")
        return ocr_result, image_data_url # Return image_data_url for Stage 2
    except Exception as e:
        return f"Initial OCR API error: {e}", None

# --- Stage 2: Validate and Enhance OCR with Image Re-comparison ---
def validate_and_enhance_ocr(client, image_data_url, initial_ocr_text, detail="high"):
    """Validates and enhances OCR text by re-comparing with the image."""
    print("\n--- STAGE 2: Validating and Enhancing OCR by Re-comparing with Image ---")

    validation_prompt = f"""
You are an expert OCR validation assistant.
You will be given an image and an initial, potentially imperfect, OCR transcription of that image.
Your task is to carefully re-examine the image and compare it against the provided transcription.
Identify and correct any errors in the transcription, including:
- Missing characters or symbols (e.g., a missing '+' or '-' sign, a digit, part of a variable).
- Incorrectly identified characters or symbols (e.g., 'l' instead of '1', 'S' instead of '5', 't' instead of '+').
- Misinterpreted mathematical structures (e.g., fraction lines, exponents, subscripts).
- Spacing issues that might affect interpretation.

Preserve the overall structure as much as possible, but prioritize accuracy of characters and symbols based on the visual evidence in the image.
The goal is to produce a more accurate transcription of exactly what is written in the image.

Here is the initial OCR transcription:
---
{initial_ocr_text}
---

Now, carefully examine the provided image and output the corrected and enhanced transcription.
"""
    try:
        response = client.chat.completions.create(
            model=GPT4O_DEPLOYMENT_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": validation_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url, "detail": detail}}
                ],
            }],
            max_tokens=2500, # Allow for potentially expanded text if fixing omissions
            temperature=0.05 # Very low temperature for high fidelity correction
        )
        enhanced_ocr_text = response.choices[0].message.content
        print("OCR Validation and Enhancement completed.")
        return enhanced_ocr_text
    except Exception as e:
        return f"OCR Validation API error: {e}"

# --- Stage 3: Final Formatting ---
def format_mathematical_solution(client, refined_text_content, problem_context=""):
    """Formats the refined mathematical text into a beautiful, step-by-step solution."""
    print("\n--- STAGE 3: Formatting the Refined Text into a Mathematical Solution ---")

    system_prompt = (
        "You are an expert mathematics professor and typesetter. Your task is to take accurately transcribed mathematical text "
        "and reformat it into a clear, elegant, step-by-step digital mathematical solution. "
        "The input text should now be a fairly accurate representation of the handwritten work."
    )

    user_prompt_parts = [
        "Please analyze the following mathematical text, which has been carefully transcribed and verified.",
        problem_context, # Optional context about the problem type
        "\nYour tasks are:",
        "1. Interpret all mathematical steps shown, including equations, expressions, and calculations.",
        "2. Reformat the entire solution into a clean, professional, step-by-step, digitally readable mathematical format. Use standard mathematical notation.",
        "3. Ensure exponents (e.g., X², sin²(x)), subscripts (e.g., x₁, a₀), fractions, roots, integrals, derivatives, summations, trigonometric functions, and other mathematical symbols are correctly and clearly represented.",
        "4. Logically structure the solution, clearly separating the problem statement (if present), variable definitions, each step of the derivation or calculation, and the final answer(s).",
        "5. Use clear visual separation for fractions (e.g., (numerator)/(denominator) or using multiple lines if the context implies a display style), and ensure alignment and readability for multi-step calculations.",
        "\nVerified Mathematical Text:",
        "---",
        refined_text_content,
        "---",
        "\nProvide the beautifully formatted mathematical solution. If the problem involves steps, show each step clearly."
    ]
    user_prompt = "\n".join(filter(None, user_prompt_parts))

    try:
        response = client.chat.completions.create(
            model=GPT4O_DEPLOYMENT_NAME, # Could be a text-focused model if preferred, but GPT-4o is fine
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=3000,
            temperature=0.2
        )
        formatted_solution = response.choices[0].message.content
        print("Final Formatting completed.")
        return formatted_solution
    except Exception as e:
        return f"Formatting API error: {e}"

# --- Dummy Image Creation (for testing) ---
def create_dummy_image_with_subtle_issues(image_path="math_problem_for_ocr.png"):
    if not os.path.exists(image_path):
        try:
            print(f"Creating a dummy image with potential OCR challenges: {image_path}...")
            img = Image.new('RGB', (700, 350), color = (240, 240, 240)) # Slightly off-white
            d = ImageDraw.Draw(img)
            try:
                # Using a slightly more "handwritten-like" font if available, or Arial
                font = ImageFont.truetype("Comic Sans MS", 28) # Example, might not be on all systems
            except IOError:
                try:
                    font = ImageFont.truetype("arial.ttf", 28)
                except IOError:
                    font = ImageFont.load_default()

            # Introduce a slight slant or imperfection if possible (harder with basic Pillow)
            # For now, focus on content that might be tricky
            lines = [
                "Solve: X² - 8X + I5 = 0", # 'I' instead of '1'
                "a=1, b=-8, c=I5",
                "X₁,₂ = -b ± √(b² - 4ac)", # Missing fraction line under this
                "         2a",             # Separated denominator
                "",
                "= -(-8) ± √((-8)² - 4(I)(I5))", # 'I' instead of '1'
                "         2(I)",
                "",
                "= 8 ± √(6Ч - 60)", # 'Ч' instead of '4'
                "     ---", # Short fraction line
                "      2",
                "",
                "X₁ = (8+2)/2 = l0/2 = S", # 'l' for '1', 'S' for '5'
                "X₂ = (8-2)/2 = 6/2 = 3"  # This one is mostly okay
            ]
            y_text = 15
            for line in lines:
                d.text((20, y_text), line, fill=(50,50,50), font=font) # Dark grey text
                y_text += 30 + (5 if line == "" else 0) # Add extra space for blank lines

            os.makedirs(os.path.dirname(image_path) or '.', exist_ok=True)
            img.save(image_path)
            print(f"Dummy image saved as '{image_path}'")
            return True
        except Exception as e:
            print(f"Could not create dummy image: {e}")
    return os.path.exists(image_path)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform multi-stage OCR and formatting for handwritten math problems using Azure OpenAI GPT-4o.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file (e.g., .png) containing the handwritten math problem.")
    parser.add_argument("--output_file", type=str, default="final_formatted_solution.txt", help="Path to save the final formatted output.")
    parser.add_argument("--context", type=str, default="", help="Optional: Brief context about the math problem type (e.g., 'This is a quadratic equation solution.').")
    parser.add_argument("--ocr_detail", type=str, default="high", choices=["low", "high", "auto"], help="Detail level for image processing during OCR stages ('low', 'high', 'auto'). 'high' is recommended for OCR.")

    args = parser.parse_args()

    azure_client = initialize_azure_client()
    if not azure_client:
        exit(1)

    # Create a dummy image if the specified one doesn't exist, for easy testing
    if not os.path.exists(args.image_path):
        print(f"Warning: Image '{args.image_path}' not found.")
        if args.image_path == "math_problem_for_ocr.png":
             if not create_dummy_image_with_subtle_issues(args.image_path):
                print(f"Failed to create or find image at '{args.image_path}'. Exiting.")
                exit(1)
        else:
            print(f"Please provide a valid image path for '--image_path'. Exiting.")
            exit(1)

    # --- STAGE 1: Initial OCR ---
    initial_ocr_text, image_data_url_for_stage2 = perform_initial_ocr(azure_client, args.image_path, detail=args.ocr_detail)
    if "Error:" in initial_ocr_text or not image_data_url_for_stage2:
        print(initial_ocr_text)
        exit(1)
    print("\n--- Initial OCR Text (Stage 1) ---")
    print(initial_ocr_text)
    with open("stage1_initial_ocr.txt", "w", encoding='utf-8') as f:
        f.write(initial_ocr_text)
    print("Saved Stage 1 OCR to stage1_initial_ocr.txt")


    # --- STAGE 2: Validate and Enhance OCR ---
    enhanced_ocr_text = validate_and_enhance_ocr(azure_client, image_data_url_for_stage2, initial_ocr_text, detail=args.ocr_detail)
    if "Error:" in enhanced_ocr_text:
        print(enhanced_ocr_text)
        exit(1)
    print("\n--- Enhanced OCR Text (Stage 2) ---")
    print(enhanced_ocr_text)
    with open("stage2_enhanced_ocr.txt", "w", encoding='utf-8') as f:
        f.write(enhanced_ocr_text)
    print("Saved Stage 2 Enhanced OCR to stage2_enhanced_ocr.txt")


    # --- STAGE 3: Final Formatting ---
    final_solution = format_mathematical_solution(azure_client, enhanced_ocr_text, args.context)
    if "Error:" in final_solution:
        print(final_solution)
        exit(1)
    print("\n\n--- Final Formatted Mathematical Solution (Stage 3) ---")
    print(final_solution)

    try:
        with open(args.output_file, "w", encoding='utf-8') as f:
            f.write(final_solution)
        print(f"\nFinal formatted solution saved to '{args.output_file}'")
    except Exception as e:
        print(f"Error saving output file '{args.output_file}': {e}")