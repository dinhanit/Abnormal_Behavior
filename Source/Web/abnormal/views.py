from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse,HttpResponse, StreamingHttpResponse
from .model_utils import model, tokenizer, model1, A
import torch
# from .SetModel import Model
# from .VniAcronym import Acronym
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
from .Pho_Chat import Chat
from django.views.decorators.csrf import csrf_exempt


import torch
import torch.nn.functional as F
import cv2
from .Inference import Inference
from .param import DEVICE


#python manage.py runsslserver --cert abnormal_certificate.crt --key abnormal.key 0.0.0.0:8000
#python manage.py runsslserver --cert abnormal_certificate.crt --key abnormal.key 192.168.1.8:8000
def analyze_sentiment1(request):
    global model1, A
    M = model1
    A = A
    if request.method == 'POST':
        input_text = request.POST['text']
        tokenized_text = A.Solve_Acr(input_text)

        label, cof = M.Predict(tokenized_text)
        with open("HisFeedBack/"+label+'.txt','a',encoding = 'utf8') as f:
            f.write(tokenized_text+'\n')
        # Update dictionary with input_text as key and label as value
        result_dict = {input_text: label}

        # Write dictionary to JSON file
        with open("HisFeedBack/hisfeedback.json", 'a', encoding='utf8') as json_file:
            json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

        with open("HisFeedBack/"+label+'.txt', 'a', encoding='utf8') as f:
            f.write(input_text+'\n')
            
        return render(request, 'abnormal/result.html', {'text': input_text, 'sentiment': label})

    response_data = {'message': 'Frame processed fail'}  # Replace with your actual response data
    return JsonResponse(response_data)


def home_page(request):
    return render(request, 'abnormal/index.html')

def members(request):
    return render(request, 'abnormal/members.html')

def feeback(request):
    return render(request, 'abnormal/feedback.html')



def feedback_analysis(request):
    with open("HisFeedBack/Positive.txt", 'r', encoding='utf8') as f:
        num_pos = len(f.readlines())
    with open("HisFeedBack/Negative.txt", 'r', encoding='utf8') as f:
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
    chatbot_response = ""

    if request.method == 'POST':
        user_input = request.POST.get('input_text', '')
        chatbot_response = Chat(user_input)  # Use your chatbot logic here
        result_dict = {user_input: chatbot_response}
        # Write dictionary to JSON file
        with open("HisFeedBack/chatbot.json", 'a', encoding='utf8') as json_file:
            json.dump(result_dict, json_file, ensure_ascii=False, indent=4)
        

    return render(request, 'abnormal/chatbot.html', {'chatbot_response': chatbot_response})


def hisfeedback(request):
    if request.method == 'GET':
        # Read the data from the hisfeedback.json file into a variable
        with open("HisFeedBack/hisfeedback.json", 'r', encoding='utf8') as json_file:
            json_data = json_file.read()

        # Clear the hisfeedback.json file by opening it in write mode
        with open("HisFeedBack/hisfeedback.json", 'w') as json_file:
            json_file.write("")

        # Create a JsonResponse with the variable's data
        response_data = {'data': json_data}  # You can add more keys to the dictionary if needed
        return JsonResponse(response_data)
    
def hischatbot(request):
    if request.method == 'GET':
        # Read the data from the chatbot.json file into a variable
        with open("HisFeedBack/chatbot.json", 'r', encoding='utf8') as json_file:
            json_data = json_file.read()

        # Clear the hisfeedback.json file by opening it in write mode
        with open("HisFeedBack/chatbot.json", 'w') as json_file:
            json_file.write("")

        # Create a JsonResponse with the variable's data
        response_data = {'data': json_data}  # You can add more keys to the dictionary if needed
        return JsonResponse(response_data)

def exam(request):
    return render(request, 'abnormal/exam.html')

@csrf_exempt
def abnormal(request):
    global model1
    if request.method == 'POST':
        frame_data = request.POST.get('frame_data')  # Get the captured frame data
        # Perform your model prediction or processing on the frame_data here
        # ...
        label = Inference(frame_data)
        print(label)

        response_data = {'label': label}  # Replace with your actual response data
        return JsonResponse(response_data)
    else:
        return JsonResponse({'message': 'Invalid request method'})
    
    
