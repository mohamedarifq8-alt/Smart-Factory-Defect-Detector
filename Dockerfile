# ==============================================================================
# Industrial AI Quality Inspection & Automated Sorting System
# Production Docker Container
# ==============================================================================
FROM python:3.11-slim

# Set environment variables for non-interactive apt and real-time logging
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DISPLAY=:0

# Install system dependencies required for OpenCV, Tkinter GUI, and V4L cameras
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    tcl8.6 \
    tk8.6 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy dependency definition first for optimal Docker layer caching
COPY requirements.txt /app/

# Upgrade pip and install Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code, model, and core application files
COPY src/ /app/src/
COPY app.py best_defect_detector.onnx /app/

# Create runtime directories for logs and defect archive
RUN mkdir -p /app/defect_records

# Expose standard Modbus TCP port for simulator or internal integration
EXPOSE 5020

# Launch industrial QC application
CMD ["python", "app.py"]
