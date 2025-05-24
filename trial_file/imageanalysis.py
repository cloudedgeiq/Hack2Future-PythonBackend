import os
import base64
import json
from openai import AzureOpenAI # Ensures you're using the OpenAI library configured for Azure
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Azure OpenAI Configuration ---
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# This will be your gpt-4o deployment name from the .env file
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"

if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME]):
    print("CRITICAL ERROR: Azure OpenAI credentials or deployment name not found in .env file.")
    print("Please ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME are set.")
    print(f"  AZURE_OPENAI_KEY: {'Set' if AZURE_OPENAI_KEY else 'NOT SET'}")
    print(f"  AZURE_OPENAI_ENDPOINT: {'Set' if AZURE_OPENAI_ENDPOINT else 'NOT SET'}")
    print(f"  AZURE_OPENAI_DEPLOYMENT_NAME: {'Set' if AZURE_OPENAI_DEPLOYMENT_NAME else 'NOT SET'}")
    exit(1)

# Initialize AzureOpenAI client
# Use a recent API version that supports gpt-4o. Check Azure docs for the latest recommended.
# "2024-02-01", "2024-03-01-preview", "2024-05-01-preview" are good candidates.
try:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-05-01-preview", # Example of a recent API version
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
except Exception as e:
    print(f"Error initializing AzureOpenAI client: {e}")
    exit(1)

# --- Image Preprocessing Functions ---
def preprocess_image(image_path, target_size=(1024, 1024), enhance=False):
    """
    Loads, preprocesses an image: resizes, converts to RGB.
    Optionally applies a sharpening filter.
    """
    try:
        img = Image.open(image_path)
        img = img.convert("RGB") # Ensure RGB format for consistency
        img.thumbnail(target_size, Image.Resampling.LANCZOS) # Resize maintaining aspect ratio

        if enhance:
            img = img.filter(ImageFilter.SHARPEN)
        return img
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def encode_image_to_base64(image_obj, format="JPEG"):
    """Encodes a PIL Image object to a base64 string."""
    if image_obj is None:
        return None
    try:
        buffered = BytesIO()
        image_obj.save(buffered, format=format, quality=85) # Added quality setting
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

# --- Azure OpenAI API Interaction ---
def analyze_maps_with_gpt_model(correct_map_b64, student_map_b64, student_image_width, student_image_height):
    """
    Sends images to the configured Azure OpenAI GPT model (e.g., gpt-4o)
    for analysis and requests bounding boxes.
    """
    if not correct_map_b64 or not student_map_b64:
        print("Error: One or both image encodings are missing for API call.")
        return None

    # This prompt is crucial. You may need to refine it for gpt-4o and your specific needs.
    prompt_text = f"""
    You are an AI assistant acting as an expert geography teacher. Your task is to evaluate a student's hand-drawn map of India.
    Image 1 is the 'correct' reference map of India, showing key geographical features such as major mountain ranges (e.g., Himalayas, Western Ghats, Eastern Ghats, Aravalli Range), major rivers, and optionally some major cities.
    Image 2 is the student's submitted map where they attempted to draw/identify these features.

    Your evaluation should consist of:
    1. A comparison of the student's map (Image 2) against the correct map (Image 1).
    2. Identification of features the student has drawn/identified correctly.
    3. Identification of features the student has drawn/identified incorrectly, or features that are significantly misplaced or missing.
    4. Brief, constructive textual feedback on the student's overall performance, highlighting strengths and areas for improvement.
    5. A suggested mark out of 10.
    6. For Image 2 (the student's map), provide precise bounding box coordinates for:
        - **Correctly drawn/identified features:** These should be marked with a green box.
        - **Incorrectly drawn, significantly misplaced, or missing features:** These should be marked with a red box. If a major feature is missing, the red box should indicate the approximate area where it *should* have been drawn.

    The student's image (Image 2) has dimensions: {student_image_width} pixels wide and {student_image_height} pixels high.
    All bounding box coordinates you provide MUST be within these dimensions [0, 0, width, height].
    Coordinates MUST be in the format [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the bounding box. Ensure x1 < x2 and y1 < y2.

    Return your response ONLY as a single, valid JSON object with the following exact structure. Do not include any text, comments, or markdown formatting (like ```json) outside of this JSON object:
    {{
      "text_feedback": "Detailed feedback on accuracy, completeness, and effort...",
      "suggested_mark": "X/10",
      "bounding_boxes": {{
        "correct": [
          {{ "feature_name": "Himalayan Range (Northern Section)", "coordinates": [50, 20, 300, 80] }},
          {{ "feature_name": "Western Ghats (Southern Part)", "coordinates": [70, 400, 120, 600] }}
        ],
        "incorrect_or_missing": [
          {{ "feature_name": "Aravalli Range", "reason": "Missing", "coordinates": [150, 150, 200, 250] }},
          {{ "feature_name": "Eastern Ghats", "reason": "Misplaced/Incomplete", "coordinates": [280, 300, 350, 500] }}
        ]
      }}
    }}
    Provide at least one example for 'correct' and one for 'incorrect_or_missing' if applicable. If all are correct, 'incorrect_or_missing' can be an empty list, and vice-versa.
    Ensure all coordinate values are integers.
    """

    try:
        print(f"Sending request to Azure OpenAI deployment: {AZURE_OPENAI_DEPLOYMENT_NAME}")
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME, # This uses your gpt-4o (or other) deployment
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{correct_map_b64}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{student_map_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=3000, # gpt-4o can handle larger contexts; adjust if response is truncated
            temperature=0.1, # Low temperature for more deterministic, structured output
            top_p=0.9,
            # response_format={"type": "json_object"} # If your API version and model support this, it's more reliable for JSON
        )

        response_content = response.choices[0].message.content
        # print(f"Raw API Response Content:\n{response_content}") # For debugging

        # Attempt to clean and parse JSON
        # Remove potential markdown ```json ... ```
        if response_content.strip().startswith("```json"):
            json_string = response_content.strip()[7:-3].strip()
        elif response_content.strip().startswith("```"):
             json_string = response_content.strip()[3:-3].strip()
        else:
            json_string = response_content.strip()

        analysis_result = json.loads(json_string)
        return analysis_result

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from API response: {e}")
        print("Problematic JSON string received from API:")
        print(json_string if 'json_string' in locals() else response_content) # Print what was attempted to be parsed
        return None
    except Exception as e:
        print(f"Error calling Azure OpenAI API or processing its response: {e}")
        # print(f"Response object: {response}") # For deeper debugging if needed
        return None

# --- Drawing Bounding Boxes ---
def draw_bounding_boxes_on_image(image_obj, analysis_result, student_image_width, student_image_height):
    """
    Draws bounding boxes on the student's image based on GPT model analysis.
    Returns the image object with drawn boxes.
    """
    if image_obj is None or analysis_result is None:
        return image_obj # Should be an image, but return if None to avoid crash

    draw = ImageDraw.Draw(image_obj)
    font_size = max(12, int(min(student_image_width, student_image_height) * 0.02)) # Dynamic font size
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size) # Common on Linux
        except IOError:
            font = ImageFont.load_default() # Fallback

    if 'bounding_boxes' not in analysis_result or not isinstance(analysis_result['bounding_boxes'], dict):
        print("Warning: 'bounding_boxes' key missing or malformed in API response. Skipping box drawing.")
        return image_obj

    def draw_single_box(coords, name, color, reason=""):
        if not (coords and len(coords) == 4 and all(isinstance(c, (int, float)) for c in coords)):
            print(f"Warning: Invalid coordinates for '{name}': {coords}. Skipping.")
            return

        # Clamp coordinates to image boundaries
        x1, y1, x2, y2 = coords
        x1 = max(0, min(x1, student_image_width -1))
        y1 = max(0, min(y1, student_image_height -1))
        x2 = max(0, min(x2, student_image_width -1))
        y2 = max(0, min(y2, student_image_height -1))
        
        # Ensure x1 < x2 and y1 < y2 after clamping
        if x1 >= x2 or y1 >= y2:
            print(f"Warning: Degenerate coordinates after clamping for '{name}': {[x1,y1,x2,y2]}. Skipping.")
            return

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{name} ({reason})" if reason else name
        
        # Adjust text position if it goes off screen
        text_x, text_y = x1, y1 - font_size - 4
        if text_y < 0: # If text goes above image
            text_y = y2 + 4 
        if text_x + draw.textlength(label, font=font) > student_image_width: # If text goes off right
            text_x = student_image_width - draw.textlength(label, font=font) - 2
        if text_x < 0: text_x = 0

        # Draw text background for better visibility
        text_bbox = draw.textbbox((text_x, text_y), label, font=font)
        draw.rectangle(text_bbox, fill=(255,255,255,180) if color != "white" else (0,0,0,180)) # Semi-transparent white/black
        draw.text((text_x, text_y), label, fill=color, font=font)


    # Draw correct boxes (green)
    for box_info in analysis_result['bounding_boxes'].get('correct', []):
        coords = box_info.get('coordinates')
        name = box_info.get('feature_name', 'Correct')
        draw_single_box(coords, name, "green")

    # Draw incorrect/missing boxes (red)
    for box_info in analysis_result['bounding_boxes'].get('incorrect_or_missing', []):
        coords = box_info.get('coordinates')
        name = box_info.get('feature_name', 'Issue')
        reason = box_info.get('reason', '')
        draw_single_box(coords, name, "red", reason)

    return image_obj

# --- Main Execution ---
if __name__ == "__main__":
    # --- !!! IMPORTANT: SET YOUR IMAGE PATHS HERE !!! ---
    correct_map_path = "Map-Actual.jpg"
    student_map_path = "Peaks and Ranges.jpg"
    # --- !!! ---

    output_annotated_filename = "student_map_annotated_gpt4o.jpg"
    output_feedback_filename = "feedback_gpt4o.txt"
    output_api_response_json = "api_response_gpt4o_debug.json"

    # Basic check for placeholder paths
    if "path/to/your" in correct_map_path or "path/to/your" in student_map_path:
        print("ERROR: Please update the placeholder paths for `correct_map_path` and `student_map_path` in the script.")
        exit(1)

    print("Step 1: Preprocessing images...")
    correct_map_pil = preprocess_image(correct_map_path)
    student_map_pil_original = preprocess_image(student_map_path, enhance=False) # Use non-enhanced for drawing

    if not correct_map_pil:
        print(f"Failed to load or preprocess the correct map: {correct_map_path}")
        exit(1)
    if not student_map_pil_original:
        print(f"Failed to load or preprocess the student map: {student_map_path}")
        exit(1)

    print("Step 2: Encoding images to Base64...")
    correct_map_b64 = encode_image_to_base64(correct_map_pil)
    student_map_b64 = encode_image_to_base64(student_map_pil_original)

    if not correct_map_b64 or not student_map_b64:
        print("Exiting due to image encoding errors.")
        exit(1)

    print("Step 3: Analyzing maps with Azure OpenAI GPT model...")
    student_img_width, student_img_height = student_map_pil_original.size
    analysis_result = analyze_maps_with_gpt_model(correct_map_b64, student_map_b64, student_img_width, student_img_height)

    if analysis_result:
        print("\n--- Analysis Result (Summary) ---")
        print(f"Text Feedback: {analysis_result.get('text_feedback', 'N/A')}")
        print(f"Suggested Mark: {analysis_result.get('suggested_mark', 'N/A')}")

        # Save full JSON response from API for debugging
        try:
            with open(output_api_response_json, "w") as f_json:
                json.dump(analysis_result, f_json, indent=2)
            print(f"Full API response JSON saved to: {output_api_response_json}")
        except Exception as e:
            print(f"Error saving API response JSON: {e}")

        print("\nStep 4: Drawing bounding boxes on student's map...")
        student_map_to_draw_on = student_map_pil_original.copy() # Draw on a copy
        annotated_image = draw_bounding_boxes_on_image(student_map_to_draw_on, analysis_result, student_img_width, student_img_height)

        if annotated_image:
            try:
                annotated_image.save(output_annotated_filename)
                print(f"Annotated student map saved to: {output_annotated_filename}")
                # annotated_image.show() # Optionally display the image
            except Exception as e:
                print(f"Error saving or showing annotated image: {e}")
        else:
            print("Failed to generate annotated image.")


        # Save textual feedback to a file
        try:
            with open(output_feedback_filename, "w", encoding="utf-8") as f_text:
                f_text.write(f"Suggested Mark: {analysis_result.get('suggested_mark', 'N/A')}\n\n")
                f_text.write("Feedback:\n")
                f_text.write(analysis_result.get('text_feedback', 'No textual feedback provided.'))
                f_text.write("\n\n--- Bounding Box Details (from API) ---\n")
                f_text.write(json.dumps(analysis_result.get('bounding_boxes', {}), indent=2))
            print(f"Textual feedback and box details saved to: {output_feedback_filename}")
        except Exception as e:
            print(f"Error saving feedback text: {e}")

    else:
        print("\nFailed to get analysis from the GPT model. Check logs and API response file (if created).")

    print("\nProcessing complete.")