FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects 7860; Render (and most other hosts) inject
# their own $PORT env var and expect the app to bind to that instead.
# Shell form (no brackets) so $PORT actually expands at container start.
EXPOSE 7860
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
