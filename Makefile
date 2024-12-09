install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

quality:
	pylint --disable=R,C *.py
	black --check *.py
	isort --check-only *py
	flake8 *py

predictor_tests:
	python -m pytest -vv ./tests/

all: install quality 
