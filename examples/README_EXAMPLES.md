# Examples Directory Structure

This directory contains organized examples demonstrating different approaches to using MLServer FastAPI Wrapper.

## Structure

### 📁 `example_titanic_manual_setup/`
**Complete manual setup example** - demonstrates the standard approach where you manually create all configuration files.

**Contents:**
- `mlserver.yaml` - Server and predictor configuration
- `mlserver_multi_classifier_simple.yaml` - Multi-classifier variant using simplified module paths
- `predictor_catboost.py` / `predictor_randomforest.py` - Custom predictor class implementations
- `train_titanic.py` / `train_titanic_2_classifiers.py` - Training scripts that create the model artifacts
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `.dockerignore` - Container build exclusions
- `artifacts/` - Pre-trained model artifacts (preprocessor, model, feature order)
- `README.md` - Detailed walkthrough of the full workflow

**Use Case:**
- When you have an existing trained model and want full control over configuration
- For production deployments where you need customized predictor logic
- Educational purposes to understand MLServer components

**Quick Start:**
```bash
cd example_titanic_manual_setup/
merve serve mlserver.yaml
```

### 📁 `example_titanic_manual_multi_classifier_setup/`
**Multi-classifier repository example** - serves multiple models (CatBoost and RandomForest) from a single repository, each independently versioned and deployable.

**Contents:**
- `mlserver_multi_classifier.yaml` - Multi-classifier configuration with per-classifier metadata, API, and build settings
- `predictor_catboost.py` / `predictor_randomforest.py` - One predictor class per classifier
- `train_titanic_2_classifiers.py` - Trains both models and writes their artifacts
- `requirements.txt` - Python dependencies
- `artifacts/` - Per-classifier artifact directories (`catboost-survival/`, `randomforest-survival/`)

**Use Case:**
- Hosting several related models in one repository
- Selecting a classifier at serve/build time with `--classifier <name>`

**Quick Start:**
```bash
cd example_titanic_manual_multi_classifier_setup/
merve serve mlserver_multi_classifier.yaml --classifier catboost-survival

# List what is available
merve list-classifiers
```

### 📁 `example_titanic_raw/`
**Raw starting point** - training script plus a Jupyter notebook with inference examples, before any MLServer configuration exists.

**Contents:**
- `train_titanic.py` - Training script that creates model artifacts
- `inference_example.ipynb` / `inference_example.py` - Inference examples showing the predictor patterns you would wrap

**Use Case:**
- Shows the typical starting state: a trained model and ad-hoc inference code
- Follow `example_titanic_manual_setup/` to see what the same project looks like after being wrapped for MLServer
- Use `merve init` to scaffold a new project skeleton (mlserver.yaml, predictor stub) as a starting point

### 📁 Root Level Files
- `mlserver_complete.yaml` - Reference configuration demonstrating all available options
- `predictor_complex.py` - Example predictor returning complex/nested response structures
- `test_complex_response.py` - Script exercising the complex response handling against a running server
- `load_test_demo.py` - Live metrics load-testing demo (see also `make demo-load`)
- `README_EXAMPLES.md` - This file

## Getting Started

### Manual Setup (single classifier)
```bash
cd examples/example_titanic_manual_setup/
merve serve
```

### Multi-Classifier Setup
```bash
cd examples/example_titanic_manual_multi_classifier_setup/
merve serve mlserver_multi_classifier.yaml --classifier catboost-survival
```

### Scaffolding a New Project
```bash
mkdir my-classifier && cd my-classifier
merve init          # generates mlserver.yaml and a predictor stub
merve validate      # check the configuration
merve serve
```

See individual README files in each subdirectory for detailed instructions.
