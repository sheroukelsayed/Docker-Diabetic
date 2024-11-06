# Use the official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to cache layer for dependencies
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app

# Expose port 5000 for the Flask app
EXPOSE 5000

# Run the Flask app when the container starts
CMD ["python", "app.py"]
