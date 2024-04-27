.PHONY: train infer

ifeq ($(OS),Windows_NT)
    detected_OS := Windows
    ACTIVATE_SCRIPT := GNN\Scripts\activate
else
    detected_OS := $(shell uname -s)
    ACTIVATE_SCRIPT := GNN/bin/activate
endif

# Define commands based on the operating system
ifeq ($(detected_OS),Windows)
    ACTIVATE_CMD := cmd /C "$(ACTIVATE_SCRIPT) &&"
else
    ACTIVATE_CMD := . $(ACTIVATE_SCRIPT) &&
endif

env:
	@echo "Creating virtual environment..."
ifeq ($(detected_OS),Windows)
	virtualenv GNN
else
	python -m venv GNN
endif
	@echo "Activating virtual environment and installing dependencies..."
	$(ACTIVATE_CMD) pip install -r requirements.txt
	@echo "Environment setup complete."


train:
	@echo "Starting training process..."
	python train.py

infer:
	@echo "Starting inference process..."
	python infer.py
