# Pinned (was the moving `4-python3.11` tag, which re-downloads the base
# whenever Microsoft publishes a new one). Same version as imageryprep.
FROM mcr.microsoft.com/azure-functions/python:4.30.0-python3.11

WORKDIR /home/site/wwwroot

# Create non-root user first so COPY --chown can reference it; Secrets dir is empty so chown is fast
RUN groupadd -r appgroup && useradd -r -g appgroup appuser \
    && mkdir -p /azure-functions-host/Secrets \
    && chown appuser:appgroup /azure-functions-host/Secrets

# Copy files with correct ownership at copy time (avoids chown -R after the fact)
# Requirements first, code after: copying hastelib before the pip install
# re-ran it (and re-downloaded every package) on any code change.
COPY --chown=appuser:appgroup api/hastefuncapi/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=appuser:appgroup hastelib/src/hastegeo /home/site/wwwroot/hastegeo
COPY --chown=appuser:appgroup hastelib/ /tmp/hastelib/
# Install hastegeo from this repo's hastelib/ (its dependencies are in
# requirements.txt). HASTE_SKIP_VERSION_BUMP avoids the haste_build.py hook
# bumping __about__.py and attempting an Azure blob upload during the build.
# Dependencies are already installed above, so this only builds and installs
# hastegeo itself (--no-deps: nothing new to download).
RUN HASTE_SKIP_VERSION_BUMP=1 pip install --no-cache-dir --no-deps /tmp/hastelib/ \
    && rm -rf /tmp/hastelib
COPY --chown=appuser:appgroup api/hastefuncapi/ .

# Convert line endings and make entrypoint executable
RUN sed -i 's/\r$//' /home/site/wwwroot/entrypoint.sh && chmod +x /home/site/wwwroot/entrypoint.sh

USER appuser

EXPOSE 8080

ENTRYPOINT ["/home/site/wwwroot/entrypoint.sh"]
