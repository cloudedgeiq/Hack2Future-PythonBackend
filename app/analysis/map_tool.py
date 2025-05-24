import os
import base64
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
import json
import re
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
GPT4O_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = "2024-05-01-preview" # Or your preferred version

llm = AzureChatOpenAI(model=GPT4O_DEPLOYMENT_NAME, api_version=AZURE_OPENAI_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY)

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
        print(f"Warning: Unknown image type for extension {ext}. Defaulting to image/jpeg.")
        print("Supported formats typically include PNG, JPEG, GIF, WEBP.")
        return "image/jpeg"

DEFAULT_EVALUATION_PROMPT_TEMPLATE = """You are an assignment evaluator. Your role is to assess student responses for school assignments ranging from 5th to 12th grade.
You will be receiving two images: the first is the student's submitted answer image, and the second is the teacher's expected answer image.
Your primary goal is to compare the student's image with the teacher's image. Specifically, check if the student's image content matches the teacher's image content and if the student has labeled all the required elements as per the assignment question, if applicable.
As the teach uploaded image is a digital image, you should focus on the visual aspects and labeling accuracy of the student's hand-drawn work in comparison to the teacher's digital image. Rather than extracting text, you will evaluate the visual and labeling aspects of the student's work.
As the student uploaded image is a hand-drawn work, so you will get to see a lot of difference in visual aspects, so check if the student labeled correctly or not.

Marks Distribution (if not otherwise specified by the question or max marks context):
- 30% of marks for overall visual matching with the teacher's image.
- 70% of marks for correct labeling and answering specific question components based on the teacher's image and the question.

You will be provided with:
- The assignment question and maximum marks.
- The student's uploaded answer (as the first image). Which is an image of their hand-draw work.
- The teacher's expected answer (as the second image). Which is an image of the digital Image.
- (Full chapter notes might be included by the user in the context below, if available. If not, evaluate based on the provided images and question.)

Your tasks are:
1. Evaluate the student's answer by analyzing the first image and comparing it to the second image and the assignment question.
2. Assign a score out of the provided 'Assignment Max Marks'. This score should reflect accuracy, completeness (including all required labels if it's a labeling task), clarity, and relevance.
3. Provide constructive feedback in a bullet-point format that is age-appropriate for the 'Student’s Class' and encourages learning.
4. Identify any missing details or misconceptions in the student's response (first image).
5. Suggest specific areas of improvement to help the student enhance their understanding.
6. Recommend 2-3 credible scholarly or educational resources (e.g., Khan Academy, JSTOR, National Geographic, or government education portals) relevant to the assignment question that can help the student.
7. Do not fabricate information. If no scholarly references are readily available or appropriate, clearly state that.
8. Ensure the credibility of all feedback and references.

Below is the user input:
Assignment Question:
{assign_que}

Student’s Class:
{student_class}

Assignment Max Marks:
{assignment_max_marks}

Instructions for output:
Please provide your entire response as a single, valid JSON object. The JSON object should have the following keys:
- "score": A numerical value representing the marks awarded.
- "feedback": A list of strings, where each string is a feedback point.
- "area_of_improvement": A list of strings, where each string suggests an area for improvement.
"""

# --- Main OCR Function ---
def ocr_with_azure_gpt4o_image(image_path_or_url,expected_output_path,assignment_max_marks,student_class,assign_que,prompt=DEFAULT_EVALUATION_PROMPT_TEMPLATE,):

    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, GPT4O_DEPLOYMENT_NAME]):
        return "Error: Azure OpenAI credentials or deployment name not configured. Please check your .env file."

    try:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

        image_data_url = ""
        original_image_data_url = ""
        if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
            image_data_url = image_path_or_url
            print(f"Using image URL: {image_data_url}")
        else:
            print(f"Using local image path: {image_path_or_url}")
            base64_image = encode_image_to_base64(image_path_or_url)
            if not base64_image:
                return "Error: Could not encode local image."
            mime_type = get_image_mime_type(image_path_or_url)
            image_data_url = f"data:{mime_type};base64,{base64_image}"
        
        if expected_output_path.startswith("http://") or expected_output_path.startswith("https://"):
            original_image_data_url = expected_output_path
            print(f"Using expected output image URL: {original_image_data_url}")
        else:
            print(f"Using expected output local image path: {expected_output_path}")
            base64_expected_image = encode_image_to_base64(expected_output_path)
            if not base64_expected_image:
                return "Error: Could not encode expected output image."
            mime_type = get_image_mime_type(expected_output_path)
            original_image_data_url = f"data:{mime_type};base64,{base64_expected_image}"

        formatted_prompt = prompt.format(
            assign_que=assign_que,
            student_class=student_class,
            assignment_max_marks=assignment_max_marks
        )

        message_content_list = [
            {"type": "text", "text": formatted_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url,
                    "detail": "high"
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": original_image_data_url,
                    "detail": "high"
                },
            },
        ]

        print("Sending request to Azure OpenAI GPT-4o...")
        response = client.chat.completions.create(
            model=GPT4O_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "user",
                    "content": message_content_list,
                }
            ],
            max_tokens=2000 
        )
        print("Received response.")
        raw_llm_output_string = response.choices[0].message.content
        print (response.choices[0].message.content)
        print(raw_llm_output_string)
        processed_evaluation_result = json.loads(re.sub(r"(^```(?:json)?\s*)|(\s*```$)", "", raw_llm_output_string.strip()).strip())
        # output_data = {
        #     "result": processed_evaluation_result,
        #     "ocr_text": raw_llm_output_string 
        # }
        return processed_evaluation_result
    
    except Exception as e:
        return f"An API error occurred: {e}"
