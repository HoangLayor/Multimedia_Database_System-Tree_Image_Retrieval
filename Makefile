.PHONY: collect index serve test evaluate clean

collect:
	python scripts/collect_data.py --limit 1000

index:
	python scripts/build_index.py

serve:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

evaluate:
	python scripts/evaluate.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -f data/features/*.npy data/features/*.index data/*.db
