# minimal Python base image
FROM python:3.11.9-slim

# metadata
LABEL version="1.0.0"
LABEL description="t2pmhc: A Structure-Informed Graph Neural Network for Predicting TCR-pMHC Binding"

# Create app directory
WORKDIR /app

# Install system build tools if needed for pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
        procps \
    && rm -rf /var/lib/apt/lists/*


# Clone repo and checkout the release tag
RUN git clone --depth 1 --branch v1.0.0 https://github.com/qbic-pipelines/t2pmhc.git /app/t2pmhc

# Set workdir
WORKDIR /app/t2pmhc

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install package
RUN pip install --no-cache-dir .

# Set entrypoint
ENTRYPOINT ["t2pmhc"]
