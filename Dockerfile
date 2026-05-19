FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose ports
EXPOSE 8000 8501

# Start both services
CMD streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 & \
    uvicorn api.main:app --host 0.0.0.0 --port 8000
