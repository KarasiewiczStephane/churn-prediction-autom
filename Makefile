.PHONY: install test lint clean run docker-build docker-run docker-shell format pre-commit

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

pre-commit:
	pre-commit run --all-files

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ coverage.xml

run:
	python -m src.cli --help

docker-build:
	docker build -t churn-predict:latest .

docker-run:
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/reports:/app/reports churn-predict:latest train

docker-shell:
	docker run --rm -it --entrypoint /bin/bash -v $(PWD):/app churn-predict:latest
