from flask import Flask, request
import json
from app.analysis.english_tool import ocr_with_azure_gpt4o_text
from app.analysis.math_tool import ocr_with_azure_gpt4o_math
from app.analysis.map_tool import ocr_with_azure_gpt4o_image


app = Flask(__name__)   

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/ocr/text', methods=['POST'])
def ocr_text():
    path = json.loads( request.get_data().decode('utf-8') )
    data = ocr_with_azure_gpt4o_text(path['path'], path['assignment_max_marks'], path['student_class'], path['assign_que'])
    return data

@app.route('/ocr/math', methods=['POST'])
def ocr_math():
    path = json.loads( request.get_data().decode('utf-8') )
    data = ocr_with_azure_gpt4o_math(path['path'], path['assignment_max_marks'], path['student_class'], path['assign_que'])
    return data

@app.route('/ocr/diagram', methods=['POST'])
def ocr_diagram():
    path = json.loads( request.get_data().decode('utf-8') )
    data = ocr_with_azure_gpt4o_image(path['path'],path['expected_output_path'] ,path['assignment_max_marks'], path['student_class'], path['assign_que'])
    return data

if __name__ == "__main__":
    app.run(debug=True)