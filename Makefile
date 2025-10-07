# OptiMOF Makefile

# Variables
EXTRACT_FEATURES = src/model_features/extract_all.py
MPI_EXTRACT_FEATURES = src/model_features/mpi_extract_all.py
COMPILER = /usr/bin/clang++
DEBUG_FLAGS = -g -Wall -fsanitize=address

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help:
	@echo "OptiMOF - Metal Organic Framework Prediction"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "  Feature Extraction:"
	@echo "    extract-all        - Extract all features (use ARGS for arguments)"
	@echo "    mpi-extract-all    - Extract features using MPI parallelization"
	@echo "    extract-feature    - Extract single feature (use ARGS for arguments)"
	@echo "    check-features     - Check extracted features"
	@echo "    select-data        - Run data selection"
	@echo ""
	@echo "  Model Training:"
	@echo "    train-thermal      - Train thermal stability model"
	@echo "    train-solvent      - Train solvent removal stability model"
	@echo "    train-water        - Train water stability model"
	@echo "    train-all          - Train all models"
	@echo ""
	@echo "  Model Testing:"
	@echo "    test-thermal       - Test thermal stability model"
	@echo "    test-solvent       - Test solvent removal stability model"
	@echo "    test-water         - Test water stability model"
	@echo "    test-all           - Test all models"
	@echo ""
	@echo "  Application:"
	@echo "    run-client         - Run Flask client or CLI (use ARGS for arguments)"
	@echo ""
	@echo "  Code Quality:"
	@echo "    format             - Format code with isort and black"
	@echo ""
	@echo "  Build:"
	@echo "    compile-zeo        - Compile zeo++ library"
	@echo "    clean-zeo          - Clean zeo++ build files"
	@echo ""
	@echo "  Cleanup:"
	@echo "    clean-all          - Remove all generated files"
	@echo "    clean-temp         - Remove temporary files"
	@echo ""
	@echo "Example usage:"
	@echo "  make train-all"
	@echo "  make run-client ARGS=\"--cif myfile.cif\""

# Feature extraction
.PHONY: extract-all
extract-all: check-features
	python $(EXTRACT_FEATURES) $(ARGS)

.PHONY: mpi-extract-all
mpi-extract-all: check-features
	python $(MPI_EXTRACT_FEATURES) $(ARGS)

.PHONY: extract-feature
extract-feature:
	python src/model_features/feature_extraction.py $(ARGS)

.PHONY: check-features
check-features:
	python data/check_features.py

.PHONY: select-data
select-data:
	python src/model_features/data_selection.py

# Model training
.PHONY: train-thermal
train-thermal:
	python src/model_training/thermal_model.py

.PHONY: train-solvent
train-solvent:
	python src/model_training/solvent_model.py

.PHONY: train-water
train-water:
	python src/model_training/water_stability_model.py

.PHONY: train-all
train-all: train-thermal train-solvent train-water

# Model testing
.PHONY: test-thermal
test-thermal:
	python src/model_training/thermal_model_test.py

.PHONY: test-solvent
test-solvent:
	python src/model_training/solvent_model_test.py

.PHONY: test-water
test-water:
	python src/model_training/water_stability_model_test.py

.PHONY: test-all
test-all: test-thermal test-solvent test-water

# MOF database for nearest neighbor query
.PHONY: run-client
run-client:
	python -m src $(ARGS)

# Code quality
.PHONY: format
format:
	@echo "Running isort to sort imports..."
	python -m isort src/ app/ data/
	@echo "Running black to format code..."
	python -m black src/ app/ data/
	@echo "Formatting complete!"

# zeo++ library
.PHONY: compile-zeo
compile-zeo: clean-zeo
	cd zeo++-0.3/voro++/src && $(MAKE)
	cd zeo++-0.3 && $(MAKE)

.PHONY: clean-zeo
clean-zeo:
	cd zeo++-0.3/voro++/src && $(MAKE) clean
	cd zeo++-0.3 && $(MAKE) clean

# Clean
.PHONY: clean-all
clean-all:
	-rm -r temp_*
	-rm data/all_in_one.pkl
	-rm data/all_in_one_cols.pkl
	-rm data/*/*_split_data.pkl
	-rm model/*.pkl
	-rm model/preprocess/*.pkl

.PHONY: clean-temp
clean-temp:
	-rm -r temp_*
