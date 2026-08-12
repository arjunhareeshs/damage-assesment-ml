# Use the official pinned TiTiler image from developmentseed
FROM developmentseed/titiler:0.18.6

# Expose the port that TiTiler runs on
EXPOSE 8000

# Command to run the TiTiler API
CMD ["uvicorn", "titiler.application.main:app", "--host", "0.0.0.0", "--port", "8000"]