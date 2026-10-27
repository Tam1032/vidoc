FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY vidoc ./vidoc
COPY data ./data
COPY evals ./evals
COPY scripts ./scripts
RUN pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1
EXPOSE 8300 8400
CMD ["uvicorn", "vidoc.api:app", "--host", "0.0.0.0", "--port", "8400"]
