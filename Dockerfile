# Backend Dockerfile - Multi-stage build with pluggable GPU support
# ==================================================================
# Build arg DEVICE controls PyTorch installation:
#   cu124  — PyTorch + CUDA 12.4 (default, ~1.2GB, for GPU machines)
#   cpu    — PyTorch CPU-only (~200MB)
#   none   — No PyTorch (lightweight ~300MB, alpha module disabled)
ARG DEVICE=cu124

# Stage 1: Install dependencies (cached unless pyproject.toml changes)
FROM python:3.14-slim AS deps

ARG DEVICE

WORKDIR /app

# Aliyun apt mirror + system build dependencies
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy ONLY dependency files — maximizes layer cache hits
COPY pyproject.toml uv.lock* ./

# Install Python dependencies based on DEVICE mode
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$DEVICE" = "none" ]; then \
      echo "==> Lightweight mode: no torch" && \
      uv pip install --system \
        --index-url https://mirrors.aliyun.com/pypi/simple/ \
        . ; \
    else \
      echo "==> GPU mode: torch with DEVICE=$DEVICE" && \
      uv pip install --system \
        --index-url https://mirrors.aliyun.com/pypi/simple/ \
        --extra-index-url https://download.pytorch.org/whl/${DEVICE} \
        --index-strategy unsafe-best-match \
        ".[gpu]" ; \
    fi

# Stage 2: Runtime image
FROM python:3.14-slim AS runtime

WORKDIR /app

# Aliyun apt mirror + minimal runtime deps
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY utils/ ./utils/
COPY session_db.py ./

# Create data directories
RUN mkdir -p data/checkpoints data/logs data/alpha_lab data/alpha_zoo

EXPOSE 8000

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-timeout", "60"]
