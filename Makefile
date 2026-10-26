build:
	docker compose build

up:
	docker compose up -d mlflow pushgateway prometheus grafana loki

down:
	docker compose down

test:
	docker compose run --rm --entrypoint pytest pipeline

prepare:
	docker compose run --rm pipeline prepare

train:
	docker compose run --rm pipeline train

simulate:
	docker compose run --rm pipeline simulate-all

causal:
	docker compose run --rm pipeline causal

.PHONY: build up down test prepare train simulate causal
