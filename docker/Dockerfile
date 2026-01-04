# Konomi Corduroy-OS - Docker Image
# Hybrid 3D tensor + LLM + XP10 OS system

FROM python:3.11-slim

LABEL maintainer="Konomi Corduroy-OS Team"
LABEL description="CPU-only AI operations with 56k dial-up protocol"
LABEL version="0.1.0"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install JAX CPU-only first (specific order matters)
RUN pip install --no-cache-dir jax[cpu]==0.4.20

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY core/ /app/core/
COPY protocol/ /app/protocol/
COPY api/ /app/api/
COPY xp10/ /app/xp10/

# Create data directory
RUN mkdir -p /app/data

# Expose ports
# 3001: API operations
# 3002: WebSocket streaming
# 6789: Cube mesh communication
EXPOSE 3001 3002 6789

# Set environment variables
ENV JAX_PLATFORM_NAME=cpu
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import jax; print('OK')" || exit 1

# Run server
CMD ["python", "-m", "protocol.server"]
