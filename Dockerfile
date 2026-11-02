# Container used to host the website on Hugging Face Spaces (free, Docker SDK).
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces sends traffic to port 7860.
EXPOSE 7860
CMD ["gunicorn", "app:server", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:7860"]