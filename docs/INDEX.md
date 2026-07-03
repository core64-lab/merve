# 📚 MLServer Documentation Index

## Quick Navigation Guide

This index provides a comprehensive overview of all documentation available for MLServer FastAPI Wrapper. Each section includes a brief description and links to detailed documentation.

---

## 📖 Core Documentation

### [API Reference](./api-reference.md)
- **REST API Endpoints**: Complete endpoint documentation with request/response schemas
- **Input Formats**: Records and ndarray payloads, opt-in auto-detection
- **Response Formats**: Prediction responses, metadata, and error handling
- **Security**: CORS configuration and request validation
- **Concurrency Control**: 503-based prediction limiting with `Retry-After`

### [Configuration Guide](./configuration.md)
- **YAML Structure**: Complete configuration schema and options
- **Server Settings**: Workers, ports, CORS, logging
- **Predictor Setup**: Module loading, initialization, and parameters
- **Observability**: Metrics, logging, correlation ID configuration
- **Multi-Classifier**: Managing multiple models in one repository

### [CLI Reference](./cli-reference.md)
- **The `mlserver` CLI**: Typer-based interface with rich output
- **Version Management**: Hierarchical tagging for complete reproducibility
- **Tag Command**: Create semantic version tags with MLServer commit tracking
- **Build Command**: Container builds with validation warnings
- **Project Setup**: `init` and `init-github` scaffolding
- **Diagnostics**: `validate`, `doctor`, `test`, `schema`
- **Command Options**: Complete parameter documentation
- **Environment Variables**: The variables MLServer actually reads
- **Examples**: Common usage patterns and workflows

### [Architecture Overview](./architecture.md)
- **System Design**: Core components and their interactions
- **Request Flow**: From API to predictor and back
- **Process Model**: Multi-worker architecture and scaling
- **Plugin System**: Dynamic predictor loading mechanism
- **Performance**: Optimization strategies

---

## 🚀 Implementation Guides

### [Deployment Guide](./deployment.md)
- **Hierarchical Versioning**: Complete reproducibility with `<classifier>-v<X.X.X>-mlserver-<hash>` tags
- **GitHub Actions Workflows**: CI/CD integration with automatic tag parsing
- **Kubernetes Strategy**: Pod-based scaling patterns with version-specific deployments
- **Container Workflow**: Building and pushing images with OCI-compliant labels
- **Multi-Classifier Deployment**: Independent versioning and deployment per classifier
- **Migration Path**: Upgrading from simple tags to hierarchical format
- **Environment Configuration**: Production settings
- **Monitoring Setup**: Prometheus and Grafana integration

### [Development Guide](./development.md)
- **Local Setup**: Development environment configuration
- **Testing Strategy**: Unit, integration, and load testing
- **Version Management**: Creating hierarchical tags during development
- **Reproducibility Testing**: Validating container builds from tags
- **Code Structure**: Package organization and conventions
- **Contributing**: Pull request guidelines with version tagging workflow
- **Debugging**: Troubleshooting common issues

### [Examples Guide](./examples.md)
- **Titanic Demo**: Complete walkthrough with CatBoost and RandomForest
- **Multi-Classifier Setup**: Deploying multiple models
- **Custom Predictors**: Writing your own predictor classes
- **Load Testing**: Performance testing with Locust
- **Monitoring Demo**: Setting up observability stack

---

## 🔧 Feature Documentation

### [Multi-Classifier Support](./multi-classifier.md)
- **Configuration Format**: Dict (canonical) and list YAML structures for multiple models
- **Deployment Patterns**: One classifier per container/process, selected at startup
- **Version Management**: Per-classifier versioning via `mlserver tag`
- **Classifier Selection**: `serve --classifier` and `MLSERVER_CLASSIFIER`
- **Examples**: Real-world multi-model deployments

### [Observability Features](./observability.md)
- **Prometheus Metrics**: Available metrics and labels
- **Structured Logging**: JSON log format and fields
- **Correlation IDs**: Request tracking across log lines
- **Grafana Dashboards**: Pre-built visualizations
- **Performance Monitoring**: Latency, throughput, errors

---

## 📋 Quick Reference Sheets

### Configuration Quick Reference
```yaml
# Minimal mlserver.yaml
server:
  port: 8000
predictor:
  module: my_predictor
  class_name: MyPredictor
```

### CLI Quick Commands
```bash
# Serve with auto-detection
mlserver serve

# Serve specific classifier
mlserver serve config.yaml --classifier model-name

# Version management with hierarchical tags
mlserver tag --classifier sentiment patch  # Create v1.0.1-mlserver-abc123
mlserver tag                               # View status of all classifiers

# Build container with version validation
mlserver build --classifier sentiment-v1.0.1-mlserver-abc123

# Version information
mlserver version --detailed

# Scaffold a new project
mlserver init --classifier my-model
```

### API Quick Endpoints
- `POST /predict` - Predictions (single and batch)
- `POST /predict_proba` - Probability predictions
- `GET /info` - Model metadata and version
- `GET /status` - Prediction slot availability
- `GET /metrics` - Prometheus metrics
- `GET /healthz` - Health check

---

## 🔍 Finding Information

### By Task
- **"I want to deploy a model"** → [Deployment Guide](./deployment.md)
- **"I want to configure the server"** → [Configuration Guide](./configuration.md)
- **"I want to write a predictor"** → [Examples Guide](./examples.md)
- **"I want to monitor performance"** → [Observability Features](./observability.md)
- **"I want to use multiple models"** → [Multi-Classifier Support](./multi-classifier.md)
- **"I want to version my classifier"** → [CLI Reference > Tag Command](./cli-reference.md#tag---version-tagging--reproducibility)
- **"I want reproducible builds"** → [Deployment Guide > CI/CD Workflow Adaptations](./deployment.md#cicd-workflow-adaptations)
- **"I want to set up CI/CD"** → [Deployment Guide > GitHub Actions](./deployment.md#3-github-actions-workflows)

### By Component
- **Server** → [Architecture Overview](./architecture.md)
- **CLI** → [CLI Reference](./cli-reference.md)
- **API** → [API Reference](./api-reference.md)
- **Config** → [Configuration Guide](./configuration.md)
- **Predictor** → [Development Guide](./development.md)

### By Problem
- **"Server won't start"** → [Development Guide > Debugging](./development.md#debugging)
- **"Predictions are slow"** → [Architecture > Performance](./architecture.md#performance-optimizations)
- **"Can't find module"** → [Configuration > Module Resolution](./configuration.md#module-resolution)
- **"Metrics not showing"** → [Observability > Troubleshooting](./observability.md#troubleshooting)

---

## 🗄️ Archive

Historical planning documents and session notes (refactoring plans, test session summaries, the never-implemented AI-init proposal, etc.) live in [`docs/archive/`](./archive/README.md). They are kept for reference only and do not describe the current system.

---

## 📝 Documentation Maintenance

This documentation is continuously updated as the codebase evolves. When making changes:

1. Update relevant documentation files
2. Update this index if new sections are added
3. Keep examples current with code changes
4. Verify links remain valid
5. Update CLAUDE.md references

**Last Updated**: 2026-07-03
**Version**: 0.3.x
