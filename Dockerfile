# Use Python 3.8 as base (compatible with your dependencies)
FROM python:3.10.4

# Install MiniZinc
RUN apt-get update && apt-get install -y \
    minizinc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command (you can override this when running the container)
# CMD ["python", "main.py"]