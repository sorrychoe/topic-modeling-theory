.PHONY: all init install install-python install-r clear

NAME = Topic Modeling Theory

SHELL := bash
python := python3

# pass pip_user_option=--user to install into the user site-packages instead
# of the (virtual) environment's default, e.g. `make install pip_user_option=--user`
pip_user_option :=

ifeq ($(OS),Windows_NT)
	python := python
endif

all:
	@echo 'the Make of Topic Modeling Theory'

init:
	$(python) -m pip install $(pip_user_option) --upgrade pip
	$(python) -m pip install jupyter

# Full setup for every notebook, including Structural_Topic_Model.ipynb (R/stm).
# Requires Rscript to be installed and on PATH.
install: install-python install-r

# Everything needed for the Python notebooks (LDA, DTM, CTM, BTM, BERTopic).
install-python:
	$(python) -m pip install $(pip_user_option) -r requirements.txt

# Only needed for Structural_Topic_Model.ipynb. Requires R/Rscript to already
# be installed on the system; skip this target if you don't plan to run that
# notebook.
install-r:
	Rscript install_package.R
	Rscript install_kernel.R

clear:
	shopt -s globstar ; \
	rm -fr **/__pycache__ **/.ipynb_checkpoints ;