# Development tasks

.PHONY: install test lint format clean

install:
	pip install -e .

test:
	pytest tests/

lint:
	flake8 src/ tests/
	isort --check src/ tests/
	black --check src/ tests/
	mypy src/ tests/

format:
	isort src/ tests/
	black src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/ .coverage
	build: clean
	python -m build