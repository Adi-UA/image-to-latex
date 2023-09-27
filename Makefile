SHELL := /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "install            : installs requirements."
	@echo "venv               : sets up virtual environment for development."
	@echo "clean              : cleans all unnecessary files and any old vm"

# Installation
.PHONY: install
install:
	python -m pip install -e . --no-cache-dir

# Set up virtual environment
.PHONY: venv
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install --upgrade pip setuptools wheel && \
	make install

# Cleaning
.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	rm -rf venv