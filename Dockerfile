# Reproducible environment for the CS2CD cheat-detection experiments.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY tests ./tests
ENV PYTHONPATH=/app/src

# Default: run the tests. Experiments:
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results cs2cd \
#       sh -c "python -m cs2cd.download -n 100 && python -m cs2cd.dataset && python -m cs2cd.run_experiments"
CMD ["pytest", "-q"]
