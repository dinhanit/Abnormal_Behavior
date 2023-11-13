from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse

import torch
import torch.nn.functional as F
import cv2
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
import random
import string

from .model_utils import sentiment_model, A
from .pho_chat import Chat

#python manage.py runsslserver --cert abnormal_certificate.crt --key abnormal.key 0.0.0.0:443
# Constants for file paths and names
FEEDBACK_HISTORY_PATH = "feedback_history/"
HISFEEDBACK_JSON_FILE = "hisfeedback.json"
CHATBOT_JSON_FILE = "chatbot.json"
EXAM_CODE_FILE = "exam_code.txt"

# Initialize abnormal count and random code
abnormal_count = 0
random_code = ""
i = 0

def analyze_sentiment1(request):
    """
    Analyze sentiment based on user input.
    """
    global sentiment_model, A
    M = sentiment_model
    A = A

    if request.method == 'POST':
        # Get input text from the POST request
        input_text = request.POST['text']
        # Tokenize the input text
        tokenized_text = A.Solve_Acr(input_text)
        # Predict sentiment label and confidence
        label, cof = M.Predict(tokenized_text)

        # Write tokenized text to feedback history
        with open(FEEDBACK_HISTORY_PATH + label + '.txt', 'a', encoding='utf8') as f:
            f.write(tokenized_text + '\n')

        # Update dictionary with input_text as key and label as value
        result_dict = {input_text: label}

        # Write dictionary to JSON file
        with open(FEEDBACK_HISTORY_PATH + HISFEEDBACK_JSON_FILE, 'a', encoding='utf8') as json_file:
            json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

        # Write input text to labeled file
        with open(FEEDBACK_HISTORY_PATH + label + '.txt', 'a', encoding='utf8') as f:
            f.write(input_text + '\n')

        return render(request, 'abnormal/result.html', {'text': input_text, 'sentiment': label})

    response_data = {'message': 'Frame processed fail'}
    return JsonResponse(response_data)


def home_page(request):
    """
    Render the home page.
    """
    return render(request, 'abnormal/index.html')


def members(request):
    """
    Render the members page.
    """
    return render(request, 'abnormal/members.html')


def feeback(request):
    """
    Render the feedback page.
    """
    return render(request, 'abnormal/feedback.html')


def feedback_analysis(request):
    """
    Analyze and render feedback data.
    """
    with open(FEEDBACK_HISTORY_PATH + "Positive.txt", 'r', encoding='utf8') as f:
        num_pos = len(f.readlines())
    with open(FEEDBACK_HISTORY_PATH + "Negative.txt", 'r', encoding='utf8') as f:
        num_neg = len(f.readlines())

    data = {
        'Category': ['Positive', 'Negative'],
        'Values': [num_pos, num_neg]
    }

    df = pd.DataFrame(data)

    categories = data['Category']
    values = data['Values']

    plt.bar(categories, values, color=['green', 'red'])
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Positive vs Negative Values')

    # Create a temporary buffer to store the plot as an image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    chart_image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return render(request, 'abnormal/feedback_analysis.html', {'chart_image': chart_image_base64})


def chatbot_view(request):
    """
    Render the chatbot page and handle user input.
    """
    chatbot_response = ""

    if request.method == 'POST':
        user_input = request.POST.get('input_text', '')
        chatbot_response = Chat(user_input)
        result_dict = {user_input: chatbot_response}

        # Write dictionary to JSON file
        with open(FEEDBACK_HISTORY_PATH + CHATBOT_JSON_FILE, 'a', encoding='utf8') as json_file:
            json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

    return render(request, 'abnormal/chatbot.html', {'chatbot_response': chatbot_response})


def hisfeedback(request):
    """
    Retrieve and clear feedback history data.
    """
    if request.method == 'GET':
        # Read the data from the hisfeedback.json file into a variable
        with open(FEEDBACK_HISTORY_PATH + HISFEEDBACK_JSON_FILE, 'r', encoding='utf8') as json_file:
            json_data = json_file.read()

        # Clear the hisfeedback.json file by opening it in write mode
        with open(FEEDBACK_HISTORY_PATH + HISFEEDBACK_JSON_FILE, 'w') as json_file:
            json_file.write("")

        # Create a JsonResponse with the variable's data
        response_data = {'data': json_data}
        return JsonResponse(response_data)


def hischatbot(request):
    """
    Retrieve and clear chatbot history data.
    """
    if request.method == 'GET':
        # Read the data from the chatbot.json file into a variable
        with open(FEEDBACK_HISTORY_PATH + CHATBOT_JSON_FILE, 'r', encoding='utf8') as json_file:
            json_data = json_file.read()

        # Clear the hisfeedback.json file by opening it in write mode
        with open(FEEDBACK_HISTORY_PATH + CHATBOT_JSON_FILE, 'w') as json_file:
            json_file.write("")

        # Create a JsonResponse with the variable's data
        response_data = {'data': json_data}
        return JsonResponse(response_data)


def exam(request):
    """
    Render the exam page with context data.
    """
    context = {
        'abnormal_url': reverse('abnormal')  # Use the reverse function to get the URL
    }
    return render(request, 'abnormal/exam.html', context)


def abnormal(request):
    """
    Process abnormal frames and update abnormal count.
    """
    global abnormal_count, random_code, i

    if request.method == 'POST':
        frame_data = request.FILES.get('frame_data')  # Get the uploaded file
        if frame_data:
            # Read the image from the uploaded file
            frame = cv2.imdecode(np.frombuffer(frame_data.read(), np.uint8), cv2.IMREAD_COLOR)
            _, image = cv2.imencode(".jpg", frame)
            url = "http://192.168.0.199:8501/process_frame/"
            files = {'file': ('image.jpg', image)}
            i += 1
            response = requests.post(url, files=files)
            label = response.text
            print(label)
            if 800 == abnormal_count:
                # Generate a random code with 5 characters
                random_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
                with open(EXAM_CODE_FILE, "w+") as f:
                    f.write(random_code)
            if label == '"Abnormal"':
                abnormal_count += 1
                print(abnormal_count)
            elif label == '"Normal"':
                abnormal_count = 0

    return JsonResponse({'abnormal_count': abnormal_count})


@csrf_exempt
def verify_code(request):
    """
    Verify entered code and reset abnormal count.
    """
    global random_code, abnormal_count
    print(random_code)

    if request.method == 'POST':
        entered_code = request.POST.get('entered_code')
        print(entered_code)
        if entered_code == random_code:
            abnormal_count = 0
            # Code is correct, implement your logic to restart camera and exam
            return JsonResponse({'code_verified': True})
        else:
            return JsonResponse({'code_verified': False})
    else:
        return JsonResponse({'code_verified': False})
