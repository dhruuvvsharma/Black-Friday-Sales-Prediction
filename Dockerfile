FROM python:3.11-slim

WORKDIR /app

# Required by XGBoost / CatBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fastapi_app/ ./fastapi_app/
COPY streamlit_app/ ./streamlit_app/
COPY src/ ./src/

# Copy trained model artifacts
COPY artifacts/model.pkl ./artifacts/model.pkl
COPY artifacts/preprocessor.pkl ./artifacts/preprocessor.pkl
COPY artifacts/frequency_maps.pkl ./artifacts/frequency_maps.pkl

ENV PYTHONPATH=/app

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Start FastAPI internally, then Streamlit publicly
CMD ["sh", "-c", "uvicorn fastapi_app.main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app/app.py --server.port 7860 --server.address 0.0.0.0"]