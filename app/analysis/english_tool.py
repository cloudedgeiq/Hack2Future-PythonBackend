import os
import base64
from openai import AzureOpenAI
# from langchain_openai import AzureChatOpenAI 
from langchain_openai import AzureChatOpenAI
import json
import re

from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
GPT4O_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = "2024-05-01-preview" # Or your preferred version

llm = AzureChatOpenAI(model=GPT4O_DEPLOYMENT_NAME, api_version=AZURE_OPENAI_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY)


from langchain_core.prompts import PromptTemplate

def llm_response(ocr_data,assignment_max_marks,student_class,assign_que):
    prompt = PromptTemplate.from_template(
    """
            You are an assignment evaluator. Your role is to assess student responses for school assignments ranging from 5th to 12th grade.
            You will be provided with:
            The assignment question and max marks to be given
            The student uploaded answer
            The teacher's expected answer
            The full chapter notes
            Your tasks are:
            Evaluate the student's answer.
            Assign a score out of max marks, based on accuracy, completeness, clarity, and relevance.
            Provide constructive feedback in a bullet-point format that is age-appropriate and encourages learning.
            Identify any missing details or misconceptions in the student's response.
            Suggest specific areas of improvement to help the student enhance their understanding.
            Recommend 2-3 credible scholarly or educational resources (e.g., Khan Academy, JSTOR, National Geographic, or government education portals) that can help the student better understand the concept.
            Do not fabricate information. If no scholarly references are available, clearly state that.
            Ensure the credibility of all feedback and references before finalizing the response.
            
            Below is the user input:
            “Assignment Question:\n”
            {assign_que}

            “Student’s Class:\n”
            {student_class}
            
            “Student’s answer:\n”
            {ocr_data}
            
            “Assignment Max Marks:\n”
            {assignment_max_marks}   
            
        
            Please provide the output in json structure strictly.
            Use the following output format:
            ## Score:
            <only scored number>
            
            ## Feedback:
            - <Point 1>
            - <Point 2>
            - <Point 3>
            (Add more points as needed)
            
            ## Area of Improvement:
            - <List specific areas or concepts the student should focus on>
            
            ## Scholarly Reference Links:
            - <Link 1>
            - <Link 2>
            - <Link 3>
            (If unavailable,     state "No credible references found.")
    """
    )
    return llm.invoke(prompt.format(ocr_data=ocr_data,assign_que=assign_que,student_class=student_class,assignment_max_marks=assignment_max_marks)).content



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
def ocr_with_azure_gpt4o_text(image_path_or_url,assignment_max_marks,student_class,assign_que,prompt="Extract all text from this image.",):

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
        raw_llm_output_string = response.choices[0].message.content
        processed_evaluation_result = llm_response(
                raw_llm_output_string,
                assignment_max_marks,
                student_class,
                assign_que
            )
        print(raw_llm_output_string)
        processed_evaluation_result = json.loads(re.sub(r"(^```(?:json)?\s*)|(\s*```$)", "", processed_evaluation_result.strip()).strip())
        output_data = {
            "result": processed_evaluation_result,
            "ocr_text": raw_llm_output_string
        }
        return output_data
    
    except Exception as e:
        return f"An API error occurred: {e}"








# --- Example Usage ---
if __name__ == "__main__":
    local_image_path = "C:\\POC\\Hack2Future\\Hack2Future-PythonBackend\\trial_file\\BadImage.jpg"
    if not os.path.exists(local_image_path) and not (local_image_path.startswith("http://") or local_image_path.startswith("https://")):
        try:
            from PIL import Image, ImageDraw, ImageFont
            print(f"'{local_image_path}' not found. Attempting to create a dummy image for testing...")
            img = Image.new('RGB', (600, 150), color = (255, 255, 255))
            d = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except IOError:
                font = ImageFont.load_default()
            d.text((10,10), "Hello Azure GPT-4o!\nThis is a test OCR image.", fill=(0,0,0), font=font)
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