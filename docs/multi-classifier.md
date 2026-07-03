# Multi-Classifier Workflow Guide

## Overview
This guide explains how to set up, develop, and deploy multiple ML classifiers from a single repository using the mlserver framework.

**Deployment model**: each classifier runs as its own server process / container serving flat endpoints (`/predict`, `/predict_proba`, ...). The classifier is selected at startup with `mlserver serve --classifier <name>` (or the `MLSERVER_CLASSIFIER` environment variable inside containers) — classifier names never appear in URLs.

## Workflow Summary

### Step 1: Train Multiple Models
Train different classifiers and save their artifacts in separate directories:

```bash
cd examples/example_titanic_manual_setup
python train_titanic_2_classifiers.py
```

This creates:
```
artifacts/
├── catboost-survival/
│   ├── model.pkl
│   ├── features.json
│   ├── metrics.json
│   └── categorical_indices.json
└── randomforest-survival/
    ├── model.pkl
    ├── features.json
    ├── metrics.json
    ├── label_encoders.pkl
    └── scaler.pkl
```

### Step 2: Create Predictor Classes
Create a predictor class for each classifier:
- `predictor_catboost_v2.py` - CatBoostSurvivalPredictor
- `predictor_randomforest.py` - RandomForestSurvivalPredictor

### Step 3: Configure Multi-Classifier YAML
Create a multi-classifier `mlserver.yaml`. The canonical format uses a **dict** under `classifiers` (keyed by classifier name):

```yaml
# Global settings (shared by all classifiers)
server:
  host: 0.0.0.0
  port: 8000

observability:
  metrics: true

repository:
  name: "titanic-multi-classifier"

# Define multiple classifiers (dict format - canonical)
classifiers:
  catboost-survival:
    classifier:
      name: "catboost-survival"
      # version auto-detected from git tags
    predictor:
      module: predictor_catboost_v2
      class_name: CatBoostSurvivalPredictor

  randomforest-survival:
    classifier:
      name: "randomforest-survival"
    predictor:
      module: predictor_randomforest
      class_name: RandomForestSurvivalPredictor

default_classifier: "catboost-survival"
```

A **list** format is also accepted and normalized to the dict format at load time (each entry must carry its name in `classifier.name`):

```yaml
classifiers:
  - classifier:
      name: "catboost-survival"
    predictor:
      module: predictor_catboost_v2
      class_name: CatBoostSurvivalPredictor
  - classifier:
      name: "randomforest-survival"
    predictor:
      module: predictor_randomforest
      class_name: RandomForestSurvivalPredictor
```

Note: `server` and `observability` are global sections — there are no per-classifier server overrides. One classifier = one server process.

### Step 4: Local Development

#### Serve specific classifier:
```bash
# Serve CatBoost classifier
mlserver serve mlserver.yaml --classifier catboost-survival

# Serve RandomForest classifier (e.g. on another port)
mlserver serve mlserver.yaml --classifier randomforest-survival --port 8001

# Serve default classifier (catboost-survival)
mlserver serve mlserver.yaml
```

#### Test endpoints (flat URLs — one classifier per server):
```bash
# The served classifier answers at /predict (no classifier name in the URL)
# Send input keys at the top level (the legacy {"payload": {...}} wrapper is deprecated)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "Pclass": 1,
      "Sex": "female",
      "Age": 25,
      "SibSp": 1,
      "Parch": 0,
      "Fare": 100.0,
      "Embarked": "S",
      "FamilySize": 2,
      "IsAlone": 0,
      "Title": "Mrs"
    }]
  }'

# The RandomForest instance started on port 8001
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [...]}'
```

### Step 5: Version Management

Use `mlserver tag` to version each classifier from git tags (the canonical version source).

```
Canonical tag format (created by `mlserver tag`): <classifier>/vX.Y.Z   (e.g. catboost-survival/v1.0.1)
Legacy tag format (still read for old tags):      <classifier>-vX.Y.Z-mlserver-<hash>
```

`mlserver tag` writes the **canonical** `<classifier>/vX.Y.Z` form; the MLServer commit lives in the annotated-tag message and the container's OCI labels, not the tag name. The **legacy** form is still parsed everywhere (status, build validation, version listing), so tags created before this change keep working.

```bash
# Create tags (auto-increments the version from the latest git tag)
mlserver tag --classifier catboost-survival patch
# ✓ Created tag: catboost-survival/v1.0.1

mlserver tag --classifier randomforest-survival minor
# ✓ Created tag: randomforest-survival/v1.1.0

# Push tags
git push --tags

# View tag status for all classifiers (add --json for machine-readable output)
mlserver tag
```

### Step 6: Container Build Strategy

#### Build Single Classifier:
```bash
# Build CatBoost classifier container
mlserver build --classifier catboost-survival

# Build an exact tagged version (validates code matches the tag)
mlserver build --classifier catboost-survival/v1.0.1
```

#### Container Naming Convention:
```
{repository}-{classifier}:{version}

Examples:
- titanic-multi-classifier-catboost-survival:1.0.1
- titanic-multi-classifier-catboost-survival:latest
- titanic-multi-classifier-randomforest-survival:1.1.0
```

(Configurable via `deployment.container_naming`.)

#### Build All Classifiers:
```bash
# Script to build all classifiers
#!/bin/bash
for classifier in catboost-survival randomforest-survival; do
  mlserver build --classifier $classifier
done
```

### Step 7: CI/CD Integration

Use `mlserver init-github` to generate a workflow triggered by tag pushes, or write your own.

> **CI trigger note:** the generated workflow (and the example below) currently trigger on the legacy pattern `'*-v*-mlserver-*'`. Since `mlserver tag` now writes canonical `<classifier>/vX.Y.Z` tags, use a `'*/v*'` trigger (and matching parsing) to fire on them — aligning the generated workflow with the canonical format is part of the ongoing tag migration.

```yaml
name: Build and Deploy Classifier

on:
  push:
    tags:
      - '*/v*'            # Canonical tag format (e.g. sentiment/v1.0.0)
      - '*-v*-mlserver-*'  # Legacy tag format (still read)

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          ref: ${{ github.ref }}

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install mlserver
        run: pip install mlserver-fastapi-wrapper

      - name: Build Container
        run: |
          mlserver build \
            --classifier ${{ github.ref_name }} \
            --registry ${{ secrets.REGISTRY_URL }}

      - name: Push Container
        run: |
          mlserver push \
            --classifier ${{ github.ref_name }} \
            --registry ${{ secrets.REGISTRY_URL }}
```

### Step 8: Kubernetes Deployment

#### Deployment per Classifier Version:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: catboost-survival-v1-0-1
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: classifier
        image: registry.example.com/titanic-multi-classifier-catboost-survival:1.0.1
        env:
        - name: MLSERVER_CLASSIFIER
          value: "catboost-survival"
---
apiVersion: v1
kind: Service
metadata:
  name: catboost-survival
spec:
  selector:
    app: catboost-survival
    version: v1.0.1
  ports:
  - port: 8000
```

Clients call the classifier through its Service at the flat path, e.g. `http://catboost-survival:8000/predict`.

### Step 9: Version Tracking

#### Response Metadata:
Every prediction response includes metadata:
```json
{
  "predictions": [1, 0, 1],
  "time_ms": 12.5,
  "predictor_class": "CatBoostSurvivalPredictor",
  "metadata": {
    "project": "titanic-multi-classifier",
    "classifier": "catboost-survival",
    "predictor_class": "CatBoostSurvivalPredictor",
    "predictor_module": "predictor_catboost_v2",
    "config_file": "mlserver.yaml",
    "git_commit": "abc123def",
    "git_tag": "catboost-survival/v1.0.1",
    "deployed_at": "2025-01-15T10:30:00Z",
    "mlserver_version": "0.3.2",
    "mlserver_api_commit": "b5dff2a",
    "mlserver_api_tag": null
  }
}
```

### Step 10: Production Workflow

1. **Development**: Train model, create predictor, test locally
2. **Tag**: `mlserver tag --classifier <name> <patch|minor|major>` creates the hierarchical git tag
3. **Push**: `git push --tags` triggers CI/CD
4. **Build**: CI/CD builds container from tag
5. **Deploy**: Deploy to dev environment
6. **Test**: Validate predictions and performance
7. **Promote**: Deploy to production
8. **Monitor**: Track metrics by classifier and version

## Best Practices

### 1. Naming Conventions
- Classifiers: `{model-type}-{purpose}` (e.g. catboost-survival)
- Git tags: canonical `{classifier}/v{semver}` (both this and the legacy `{classifier}-v{semver}-mlserver-{hash}` form are read; `mlserver tag` currently writes the legacy form)
- Containers: `{repo}-{classifier}:{version}`

### 2. Version Management
- Each classifier has independent versioning
- Use semantic versioning (major.minor.patch) via `mlserver tag`
- Tag every production deployment
- Versions are derived from git tags — no manual version fields in the config

### 3. Testing Strategy
- Unit tests per predictor class
- Integration tests per classifier
- Load testing against each classifier's deployment
- A/B testing with multiple versions behind a service mesh

### 4. Deployment Strategy
- Immutable containers (never update, always redeploy)
- Blue-green deployments per classifier
- Canary deployments for new versions
- Separate scaling policies per classifier

## Migration from Single to Multi-Classifier

### For Existing Projects:
1. Keep existing `mlserver.yaml` for backward compatibility
2. Move the per-classifier sections (`predictor`, `classifier`, `api`) under a named entry in `classifiers:`
3. Update CI/CD to pass `--classifier` to build/push
4. Gradually migrate to multi-classifier structure

### Detection Logic:
```python
# The CLI automatically detects configuration type:
if "classifiers" in config:
    # Multi-classifier mode
    use_multi_classifier_logic()
else:
    # Single classifier mode (backward compatible)
    use_single_classifier_logic()
```

## Troubleshooting

### Issue: "No classifier specified"
**Solution**: Either specify `--classifier <name>` or set `default_classifier` in config

### Issue: Container build fails
**Solution**: Ensure predictor module paths are correct and artifacts exist

### Issue: Wrong version deployed
**Solution**: Check git tag matches container tag and deployment manifest

### Issue: Performance degradation
**Solution**: Monitor metrics per classifier deployment, scale replicas independently

## Example Commands Reference

```bash
# Local development
mlserver serve mlserver.yaml --classifier catboost-survival

# List available classifiers
mlserver list-classifiers mlserver.yaml

# Tag a release (canonical <classifier>/vX.Y.Z tag)
mlserver tag --classifier catboost-survival patch

# Build specific version
git checkout catboost-survival/v1.2.3
mlserver build --classifier catboost-survival

# Push to registry
mlserver push --classifier catboost-survival --registry gcr.io/myproject

# Check version info
mlserver version --classifier catboost-survival

# Run tests for specific classifier
pytest tests/test_catboost_predictor.py
```

## Next Steps

1. Implement automated version bumping script
2. Add classifier-specific health checks
3. Create Grafana dashboards per classifier
4. Implement automated rollback on metric degradation
5. Add support for ensemble predictions across classifiers
