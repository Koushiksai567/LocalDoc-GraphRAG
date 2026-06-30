.PHONY: install dev bootstrap up dependencies down reset logs api ingest ask test lint typecheck check eval clean

install:
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install -e .

dev:
	python -m pip install -r requirements-dev.txt

bootstrap:
	./scripts/bootstrap_local.sh

up:
	docker compose up --build -d

dependencies:
	docker compose up -d neo4j

down:
	docker compose down

reset:
	docker compose down -v

logs:
	docker compose logs -f model-init ollama neo4j api

api:
	uvicorn enterprise_graphrag.main:app --host 0.0.0.0 --port 8000

ingest:
	curl -fsS -X POST http://localhost:8000/v1/ingest \
	  -H 'Content-Type: application/json' \
	  -d '{"path":"sample_data","recursive":true,"replace_existing":true}' | python -m json.tool

ask:
	curl -fsS -X POST http://localhost:8000/v1/query \
	  -H 'Content-Type: application/json' \
	  -d '{"query":"Why can checkout fail after inventory committed?"}' | python -m json.tool

test:
	pytest -q

lint:
	ruff check src tests

typecheck:
	mypy src

check: lint typecheck test
	python -m compileall -q src tests

eval:
	python -m enterprise_graphrag.evaluation --dataset eval/evalset.jsonl --output artifacts/evaluation

clean:
	rm -rf artifacts .pytest_cache .mypy_cache .ruff_cache .model_cache
