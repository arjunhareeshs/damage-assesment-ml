# Use the official Azurite image from Microsoft
# Pinned to what `latest` pointed to on 2026-11-05, so the image only
# changes when this line does.
FROM mcr.microsoft.com/azure-storage/azurite:3.37.0

# Expose the necessary ports
EXPOSE 10000 10001 10002

# Command to run Azurite with --skipApiVersionCheck to handle newer SDK versions
CMD ["azurite", "-l", "/data", "--blobHost", "0.0.0.0", "--queueHost", "0.0.0.0", "--tableHost", "0.0.0.0", "--skipApiVersionCheck"]