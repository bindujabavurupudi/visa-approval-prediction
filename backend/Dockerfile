# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code from app/ into the container's /app
COPY app/ ./app/
COPY model/ ./model/

# Expose the port the app runs on
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.app:app
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask app
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
