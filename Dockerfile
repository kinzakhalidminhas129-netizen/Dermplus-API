FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port 7860 (Default for HF)
EXPOSE 7860

# Use Gunicorn for production instead of app.run()
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]