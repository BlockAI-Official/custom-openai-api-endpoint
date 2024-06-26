# Use an official Python runtime as a parent image
FROM python:3.12.3-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        curl \
        build-essential \
        python3-dev \
        libffi-dev \
        libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Ensure that Poetry installs to system
ENV PATH="/root/.local/bin:$PATH"

# Set work directory
WORKDIR /app

# Copy the project files into the working directory
COPY . /app

# Initialize Node.js project and install @solana/web3.js
RUN npm init -y \
    && npm install @solana/web3.js

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --verbose

# Install python-dotenv
RUN pip install python-dotenv

# Expose the port the app runs on (using default value here)
EXPOSE ${PORT}

# Command to run the application, ensuring it loads the environment variables
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
