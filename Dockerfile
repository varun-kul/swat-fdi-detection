FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy source
COPY dataset.py model.py threshold.py infer.py ./

# Copy model artifacts (committed to repo or downloaded at startup)
COPY best_model.pt scaler.pkl threshold.pkl ./

EXPOSE 8000

CMD ["uvicorn", "infer:app", "--host", "0.0.0.0", "--port", "8000"]