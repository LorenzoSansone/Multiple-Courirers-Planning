FROM mcp-base

# Set the Python path to include both SMT directory and root
ENV PYTHONPATH=/app:/app/SMT

# Ensure the results directory exists
RUN mkdir -p /app/res/SMT

# Set working directory
WORKDIR /app

ENTRYPOINT ["python", "SMT/smtsolver.py"]
CMD ["--help"] 