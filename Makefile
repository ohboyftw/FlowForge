.PHONY: install test smoke lint format examples clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

smoke:
	python -m pytest tests/test_smoke.py -v

test-cov:
	python -m pytest tests/ -v --cov=flowforge --cov-report=term-missing

lint:
	ruff check src/ tests/ examples/

format:
	ruff format src/ tests/ examples/

examples:
	@echo "Running all examples..."
	python examples/01_research_report.py
	@echo ""
	python examples/02_customer_support.py
	@echo ""
	python examples/03_content_pipeline.py
	@echo ""
	python examples/04_stock_analysis.py
	@echo ""
	python examples/05_tango_review.py
	@echo "\nAll examples complete."

typecheck:
	mypy src/flowforge/

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

build:
	python -m build

publish-test:
	twine upload --repository testpypi dist/*

publish:
	twine upload dist/*
