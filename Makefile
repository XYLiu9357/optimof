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
	@echo "    preprocess         - Run data preprocessing"
	@echo ""
	@echo "  Hyperparameter Tuning:"
	@echo "    tune-thermal       - Tune thermal stability model (use TRIALS=N)"
	@echo "    tune-solvent       - Tune solvent removal stability model (use TRIALS=N)"
	@echo "    tune-water         - Tune water stability model (use TRIALS=N)"
	@echo "    tune-all           - Tune all models"
	@echo ""
	@echo "  Model Training:"
	@echo "    train-thermal      - Train thermal stability model (use STRUCTURE=path)"
	@echo "    train-solvent      - Train solvent removal stability model (use STRUCTURE=path)"
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
	@echo "    build-mof-map      - Build MOF map for nearest neighbor queries"
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
	python -m src.model_features.extract_all $(ARGS)

.PHONY: mpi-extract-all
mpi-extract-all: check-features
	python -m src.model_features.mpi_extract_all $(ARGS)

.PHONY: extract-feature
extract-feature:
	python -m src.model_features.feature_extraction $(ARGS)

.PHONY: check-features
check-features:
	python -m data.check_features

.PHONY: preprocess
preprocess:
	python -m src.model_features.preprocess

# Hyperparameter tuning
.PHONY: tune-thermal
tune-thermal:
	python -m src.model_training.tune_thermal --n-trials $(or $(TRIALS),50)

.PHONY: tune-solvent
tune-solvent:
	python -m src.model_training.tune_solvent --n-trials $(or $(TRIALS),50)

.PHONY: tune-water
tune-water:
	python -m src.model_training.tune_water --n-trials $(or $(TRIALS),50)

.PHONY: tune-all
tune-all: tune-thermal tune-solvent tune-water

# # Model training
# .PHONY: train-thermal
# train-thermal:
# ifdef STRUCTURE
# 	python -m src.model_training.thermal_model --structure $(STRUCTURE)
# else
# 	python -m src.model_training.thermal_model
# endif
# 
# .PHONY: train-solvent
# train-solvent:
# ifdef STRUCTURE
# 	python -m src.model_training.solvent_model --structure $(STRUCTURE)
# else
# 	python -m src.model_training.solvent_model
# endif
# 
# .PHONY: train-water
# train-water:
# 	python -m src.model_training.water_stability_model
# 
# .PHONY: train-all
# train-all: train-thermal train-solvent train-water
 
# Model testing
.PHONY: test-thermal
test-thermal:
	python -m src.model_training.tests.test_thermal

.PHONY: test-solvent
test-solvent:
	python -m src.model_training.tests.test_solvent

.PHONY: test-water
test-water:
	python -m src.model_training.tests.test_water

.PHONY: test-all
test-all: test-thermal test-solvent test-water

# MOF database for nearest neighbor query
.PHONY: build-mof-map
build-mof-map:
	python -m src.utils.build_mof_map

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
	-rm -rf temp_*
	-rm -f data/all_in_one.pkl
	-rm -f data/*/*_clean_data.pkl
	-rm -f data/*/*_test_data.pkl
	-rm -f model/*.pkl
	-rm -f model/scalers/*.pkl
	-rm -f model/structure/*_structure.json

.PHONY: clean-temp
clean-temp:
	-rm -r temp_*
