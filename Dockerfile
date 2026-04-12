FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir . # Install project from pyproject.toml
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose HuggingFace Space port
EXPOSE 7860

# Use the installed script entry point
CMD ["server"]
