DC=docker compose

.PHONY: up down init seed graph collect ingest-energy ops-schema cache-schema vector-index fact-graph-schema job-schema warmup worker test regression api frontend frontend-build

up:
	$(DC) up -d

down:
	$(DC) down

init:
	python3 scripts/bootstrap_db.py

seed:
	python3 scripts/seed_sample_data.py

graph:
	python3 scripts/extract_graph.py

collect:
	python3 scripts/collect_energy_data.py

ingest-energy:
	python3 scripts/ingest_energy_data.py

ops-schema:
	psql "$$DATABASE_URL" -f sql/006_operational_hardening.sql

cache-schema:
	psql "$$DATABASE_URL" -f sql/007_rag_caches.sql

vector-index:
	psql "$$DATABASE_URL" -f sql/008_vector_indexes.sql

fact-graph-schema:
	psql "$$DATABASE_URL" -f sql/009_research_fact_graph.sql

job-schema:
	psql "$$DATABASE_URL" -f sql/010_async_job_queue.sql

warmup:
	python3 scripts/warmup_rag.py

worker:
	python3 scripts/job_worker.py

test:
	python3 -m pytest tests -q

regression:
	python3 scripts/regression_check.py

api:
	uvicorn rag_assistant.api.app:app --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build
