FROM python:3.12-slim

# prometheus-flask-exporter e exigido por `mlflow server --expose-prometheus`.
# sqlalchemy fixado: o mlflow 2.17 nao importa com SQLAlchemy >= 2.1 (FallbackAsyncAdaptedQueuePool).
RUN pip install --no-cache-dir mlflow==2.17.2 prometheus-flask-exporter==0.23.1     "sqlalchemy==2.0.36"

RUN mkdir -p /mlflow/artifacts /mlflow/metrics
WORKDIR /mlflow
VOLUME /mlflow
EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts", \
     "--expose-prometheus", "/mlflow/metrics"]
