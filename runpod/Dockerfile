FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY handler.py .
COPY crqa_endpoint.py .
CMD ["python", "handler.py"]
