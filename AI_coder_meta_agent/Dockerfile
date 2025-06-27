FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set workdir to project root
WORKDIR /app

# Install dependencies from AI_coder_meta_agent directory
COPY ./AI_coder_meta_agent/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Add project root to Python path
ENV PYTHONPATH=/app:$PYTHONPATH

EXPOSE 5000 5001 5002 5003 5004 5006 5007