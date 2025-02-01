FROM mcp-base

# Set the Python path to include both SAT directory and root
ENV PYTHONPATH=/app:/app/SAT

# Ensure the results directory exists
RUN mkdir -p /app/res/SAT

# Set working directory
WORKDIR /app/SAT


ENTRYPOINT ["python", "./model_sat.py"]