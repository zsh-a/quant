# Backend Dockerfile - Optimized with UV and pyproject.toml
FROM python:3.14-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH (install script puts it in ~/.local/bin on Unix)
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency file
COPY pyproject.toml ./

# Install Python dependencies with uv (much faster than pip)
# Use --no-cache to reduce image size
RUN uv pip install --system --no-cache .

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/checkpoints data/logs

# Expose port
EXPOSE 8000

# Default command (can be overridden)
# --ws-ping-timeout 60: avoid closing connections when client is slow to pong (e.g. tab in background)
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-timeout", "60"]
