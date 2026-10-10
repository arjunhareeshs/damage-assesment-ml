# Use the official pinned TiTiler image from developmentseed. Recent
# releases (>=0.4) are published to GHCR, not Docker Hub -- the
# docker.io/developmentseed/titiler repo only has tags up to 0.4.0a2/latest
# (both last pushed 2021), so a Hub-style "developmentseed/titiler:0.18.6"
# reference 404s.
FROM ghcr.io/developmentseed/titiler:0.18.6

# Expose the port that TiTiler runs on
EXPOSE 8000

# Command to run the TiTiler API
CMD ["uvicorn", "titiler.application.main:app", "--host", "0.0.0.0", "--port", "8000"]