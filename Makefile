install:
	pip install -e ".[dev]"

run-ui:
	streamlit run streamlit_app.py

run-api:
	uvicorn graphrag_studio.api:app --reload

ingest-sample:
	python scripts/ingest_sample_corpus.py

benchmark:
	python scripts/run_benchmark.py --cases data/benchmark/heldout_queries.json

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

test:
	pytest -q
