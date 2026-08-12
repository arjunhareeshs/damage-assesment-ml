# This container only uploads a few small JSON blobs into Azurite, so it does
# not need the ~2.4 GB devcontainers image (full toolchain, git, node, zsh,
# oh-my-zsh, debuggers). python:3.11-slim is ~150 MB and runs the same script.
FROM python:3.11-slim

# Install Azure Storage Blob SDK
RUN pip install --no-cache-dir azure-storage-blob azure-storage-queue

# Copy the data files from setup folder to the container
COPY ./setup/config_admin_settings.json /data/config_admin_settings.json

# Create placeholder files for users_acl and project_stats if they don't exist in your repo
# users_acl.json should be an array of user objects - empty array, users auto-created in dev mode
RUN echo '[]' > /data/users_acl.json
RUN echo '{"projects": [], "project_count": 0, "layer_count": 0, "model_count": 0}' > /data/project_stats.json

# Copy the upload script
COPY ./docker/data-init/upload_data.py /upload_data.py

# Make the script executable
RUN chmod +x /upload_data.py

# Run the upload script
CMD ["python", "/upload_data.py"]
