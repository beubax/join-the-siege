# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies required by python-magic and potentially other libs
# libmagic-dev is common for python-magic
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY ./src /app/src/
COPY ./data /app/data/

# Expose port (same as Uvicorn default)
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Entrypoint script could be useful for more complex startup logic e.g. waiting for DB
# For now, direct uvicorn command is fine. If running with docker-compose, 
# app service can depend_on db service.

# Note for Celery worker: It would run in a separate container but use much of the same base. 