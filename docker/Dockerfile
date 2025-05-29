FROM python:3.11-slim

# Set working directory
WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY flask_server.py .
COPY face_logic.py .
COPY train_user_model.py .
COPY verify_user.py .

# Expose the Flask port
EXPOSE 5000

# Run the Flask server
CMD ["python", "flask_server.py"]