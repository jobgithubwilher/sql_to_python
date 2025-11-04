# Use a lightweight Python base image suitable for production containers
FROM python:3.11-slim

# Ensure logs stream directly to stdout/stderr and pip caches are disabled
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Establish the application directory inside the container
WORKDIR /app

# Copy the project files into the image (Docker layer caching keeps rebuilds fast)
COPY . /app

# Install Python dependencies needed by the Streamlit app
RUN pip install --upgrade pip && \
    pip install pandas streamlit

# Streamlit listens on port 8501 by default
EXPOSE 8501

# Launch the Streamlit web app when the container starts
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
