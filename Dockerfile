# minimal Python base image
FROM python:3.11.9-slim

# metadata
LABEL version="0.1.0"
LABEL description="t2pmhc: A Structure-Informed Graph Neural Network for Predicting TCR-pMHC Binding"

# Create app directory
WORKDIR /app

# Install system build tools if needed for pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# Download the GitHub release (replace user/repo)
#RUN wget https://github.com/qbic-pipelines/t2pmhc/archive/refs/tags/0.1.0.tar.gz -O source.tar.gz

# Extract the archive
#RUN tar -xzf source.tar.gz && \
#    rm source.tar.gz

# test it
COPY . /app


# Move into the extracted folder (GitHub exports as repo-<tag>)
#WORKDIR /app/t2pmhc-0.1.0

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install package
RUN pip install --no-cache-dir .

# Set entrypoint
ENTRYPOINT ["t2pmhc"]
