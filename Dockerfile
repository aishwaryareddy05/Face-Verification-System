FROM python:3.10-bookworm

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_SYSTEM_PYTHON=1

# Enable testing repo temporarily for newer libstdc++6
RUN echo "deb http://deb.debian.org/debian testing main" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get -t testing install -y libstdc++6 && \
    sed -i '$d' /etc/apt/sources.list && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install system dependencies for OpenCV, cv2, and others
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    g++ \
    wget \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Install Python dependencies with uv
COPY requirements.txt .
COPY wheels /app/wheels
RUN pip install --no-index --find-links=/app/wheels -r requirements.txt

# Copy application code
COPY . .

# Create model and log directories
RUN mkdir -p /app/models /app/logs /home/appuser/.insightface/models && \
    chown -R appuser:appuser /app /home/appuser

# Copy model download script
COPY download_models.py /tmp/download_models.py

# Optional: Uncomment to download models at build time
# RUN python /tmp/download_models.py /app/models && rm /tmp/download_models.py

# or copy pre-downloaded models
COPY ./models/ /app/models/

# Set model permissions
RUN chown -R appuser:appuser /app/models

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
