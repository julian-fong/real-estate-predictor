install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

quality:
	pylint --disable=R,C *.py
	black *.py
	isort *py
	flake8 *py

all: install quality 
