FROM python:3.11-slim

WORKDIR /app

# CPU-only image. Generation runs in a separate GPU service; the API starts, serves and is
# tested without torch present.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY artifacts/ ./artifacts/

ENV MODEL_BACKEND=stub
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
