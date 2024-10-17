# Use the official Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory to /app
WORKDIR /app

# Install the required packages
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    curl \
    gcc \
    libssl-dev \
    && apt-get clean

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add the Poetry's bin to the PATH
ENV PATH="/root/.local/bin:$PATH"

# Install the dependencies
RUN poetry install --no-root

# Copy the current directory contents into the container at /app
COPY . /app

# Run the command
CMD ["poetry", "run", "python", "main.py"]