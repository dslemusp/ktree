PROJECT:= r002_execution

.PHONY: format isort lint typing

check: format isort lint typing

format:
	black .

isort:
	isort .

lint:
	-flake8 .

typing:
	-mypy .