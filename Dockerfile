# Use the official Python 3.13 slim image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /workspace

# Install system utilities needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install dependencies (using --no-cache-dir to minimize image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code and model artifacts
COPY app/ app/
COPY src/ src/
COPY models/ models/
COPY data/ data/

# Expose the API port
EXPOSE 8000

# Start the FastAPI application with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
