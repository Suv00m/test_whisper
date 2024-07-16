# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whisper.py file to the container
COPY whisper.py .

# Set the command to run when the container starts
CMD ["python", "whisper.py"]