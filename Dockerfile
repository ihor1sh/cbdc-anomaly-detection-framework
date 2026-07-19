# ============================================================================
# CBDC ML Framework - Python environment for Article 2.1 experiments
# ============================================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies for scientific Python
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1. PyTorch CPU-only (smaller image, laptop-friendly)
RUN pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# 2. Everything else from standard PyPI (DGL removed - GCN uses pure PyTorch)
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Pre-create data and results directories
RUN mkdir -p /workspace/data /workspace/results /workspace/notebooks /workspace/src

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
