# Use an appropriate base image, e.g., python:3.10-slim
FROM python:3.11-slim

# Set environment variables (e.g., set Python to run in unbuffered mode)
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /PerunaBot-0.01

# Copy your application's requirements and install them
COPY requirements.txt ./PerunaBot-0.01/

RUN pip install -r ./PerunaBot-0.01/requirements.txt

# Copy your application code into the container
COPY . /PerunaBot-0.01/

EXPOSE 8080

CMD ["python", "-m", "chainlit", "run", "hosted-app.py", "-h", "--port", "8080"]