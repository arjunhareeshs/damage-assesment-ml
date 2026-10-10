# Use the official stable Azure Functions Python 3.11 slim image
#
# Build from the REPO ROOT (haste-fork/), same as every other image in this
# docker/ directory -- COPY paths below are relative to that context, not to
# this Dockerfile's own directory (Docker always resolves COPY against the
# build context root; `../../` paths reaching outside the context are
# rejected outright, which is what this file did before this fix):
#   az acr build --registry <name> --image hasteapi:latest --file docker/api/Dockerfile .
FROM mcr.microsoft.com/azure-functions/python:4.30.0-python3.11-slim

# Set the working directory
WORKDIR /home/site/wwwroot

# Copy requirements first to leverage Docker layer caching
COPY api/hastefuncapi/requirements.txt .

# Install dependencies with no-cache-dir to minimize image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Azure Functions API application code
COPY api/hastefuncapi/ .

# Expose the Azure Functions host port
EXPOSE 7071

# Start the Azure Functions host
CMD ["func", "host", "start"]