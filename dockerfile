# official Python image as the base image
FROM python:3.9-slim

# Setting environment variables
ENV PYTHONUNBUFFERED 1

# working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . /app

# Specify the command to run the main script
CMD ["python", "main.py"]
