FROM python:3.10

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx  # This library provides libGL.so.1
    
WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["python","FApi.py"]

