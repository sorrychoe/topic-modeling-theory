.PHONY: init install clear

NAME = Topic Modeling Theory

SHELL := bash
python := python3

ifeq ($(OS),Windows_NT)
	python := python
endif

all:
	@echo 'the Make of Topic Modeling Theory'

init:
	$(python) -m pip install $(pip_user_option) --upgrade pip
	$(python) -m pip install jupyter

install:
	$(python) -m pip install $(pip_user_option) -r requirements.txt
	Rscript lib/install_package.R
	Rscript lib/install_kernel.R

clear:
	shopt -s globstar ; \
	rm -fr **/__pycache__ **/.ipynb_checkpoints ;