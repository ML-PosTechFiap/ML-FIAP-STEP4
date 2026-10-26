FROM python:3.12-slim

# libgomp1 e necessario para o xgboost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pyproject + src antes do install: o modo editavel aponta para /app/src,
# o mesmo caminho que o bind mount do compose sobrepoe depois.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .[dev]

ENTRYPOINT ["python", "-m", "credit_monitor"]
