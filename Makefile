.PHONY: all install quality predictor_tests

check_dirs := real_estate_predictor real_estate_predictor/config real_estate_predictor/datasets real_estate_predictor/models real_estate_predictor/processing real_estate_predictor/tests real_estate_predictor/utils

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

quality:
	pylint --disable=R,C $(check_dirs)/*.py
	black --check $(check_dirs)/*.py
	isort --check-only $(check_dirs)/*.py
	flake8 $(check_dirs)/*.py

black_full: 
	black $(check_dirs)/*.py

predictor_tests:
	python -m pytest -vv real_estate_predictor/tests/

all: install quality 
