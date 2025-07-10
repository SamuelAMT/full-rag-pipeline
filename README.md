# Full RAG Pipeline

## Project Overview
A comprehensive RAG (Retrieval-Augmented Generation) pipeline built with LangGraph, featuring document processing, vector storage, retrieval, and generation with monitoring and safety measures.

## Architecture
```
User Query → Document Processing → Vector Storage → Retrieval → LLM Generation → Response
     ↓              ↓                   ↓             ↓               ↓             ↓
  FastAPI      Unstructured         ChromaDB      LangGraph         llama       Monitoring
```

## Project Structure
```
full-rag-pipeline/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── infra/
│   ├── terraform/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── database.py
│   │   └── logging.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── processing.py
│   │   └── synthetic_data.py
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── faiss_store.py
│   │   ├── postgres_store.py
│   │   └── embeddings.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retrievers.py
│   │   └── rerankers.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_models.py
│   │   └── prompts.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── nodes.py
│   │   ├── edges.py
│   │   └── workflow.py
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── content_filter.py
│   │   └── guardrails.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── logging.py
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       ├── routes.py
│       └── schemas.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_graph.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_evaluation.ipynb
│   └── 03_model_evaluation.ipynb
├── scripts/
│   ├── setup_data.py
│   ├── run_pipeline.py
│   └── evaluate_model.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
├── Dockerfile
└── docker-compose.yml
```

## Technology Stack

### Core Framework
- **LangGraph**: Workflow orchestration and state management
- **LangChain**: LLM integration and RAG components
- **FastAPI**: REST API framework
- **Pydantic**: Data validation and settings management

### LLM & Embeddings
- **Primary**: llama (free, local deployment)
- **Alternative**: OpenAI GPT (with API key)
- **Embeddings**: sentence-transformers (free) or OpenAI embeddings

### Vector Stores
- **ChromaDB**: Local vector storage

### Document Processing
- **Unstructured**: Document parsing and chunking
- **PyPDF2**: PDF processing

### Monitoring & Safety
- **Prometheus + Grafana**: Metrics and monitoring (free)
- **LangSmith**: LangChain monitoring (free tier)
- **Guardrails**: Content safety filters

### Infrastructure
- **Docker**: Containerization
- **Terraform**: Infrastructure as Code
- **GitHub Actions**: CI/CD pipeline
- **Poetry**: Dependency management

## Key Features

### 1. Document Ingestion Pipeline
- Support for PDF, DOCX, TXT, and web scraping
- Intelligent chunking with overlap
- Metadata extraction and preservation

### 2. Vector Storage & Retrieval
- Multiple vector store backends
- Hybrid search (semantic + keyword)
- Reranking for improved relevance

### 3. LangGraph Workflow
- Multi-step reasoning
- Conditional routing
- Error handling and retries
- State persistence

### 4. Generation Pipeline
- Prompt engineering with templates
- Response streaming
- Context window management
- Response validation

### 5. Safety & Monitoring
- Content filtering
- Response quality scoring
- Performance metrics
- Request/response logging

## Quick Start

### Prerequisites
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

### Setup
```bash
# Clone repository
git clone https://github.com/SamuelAMT/full-rag-pipeline.git
cd full-rag-pipeline

# Install dependencies
poetry install

# Download and install llama model
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" --local-dir ./models

# Setup environment
cp .env.example .env

# Download sample data
python scripts/setup_data.py

# Start services
docker-compose up -d

# Run the application
poetry run uvicorn src.api.main:app --reload
```

### Usage
```bash
# Health check
curl http://localhost:8000/health

# Upload documents
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Query the RAG system
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of the document?"}'
```

## Environment Variables
```env
# LLM Configuration
LLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=openai_key_here

# Vector Store
VECTOR_STORE_TYPE=chromadb

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Monitoring
LANGSMITH_API_KEY=langsmith_key
LANGSMITH_PROJECT=full-rag-pipeline
```

## Development Workflow

### 1. Local Development
```bash
# Start development environment
poetry shell
docker-compose up -d postgres redis

# Run tests
poetry run pytest

# Run with hot reload
poetry run uvicorn src.api.main:app --reload
```

### 2. Testing
```bash
# Unit tests
poetry run pytest tests/

# Integration tests
poetry run pytest tests/integration/

# Performance tests
poetry run pytest tests/performance/
```

### 3. Deployment
```bash
# Build and push Docker image
docker build -t samuelamt/full-rag-pipeline:latest .
docker push samuelamt/full-rag-pipeline:latest

# Deploy with Terraform
cd infra/terraform
terraform init
terraform plan
terraform apply
```

## Model Evaluation

### Retrieval Metrics
- **Precision@K**: Relevance of top K retrieved documents
- **Recall@K**: Coverage of relevant documents in top K
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain

### Generation Metrics
- **BLEU**: N-gram overlap with reference
- **ROUGE**: Summary quality metrics
- **BERTScore**: Semantic similarity
- **Custom Relevance**: Domain-specific scoring

## Monitoring Dashboard

### Metrics Tracked
- Request/response times
- Token usage and costs
- Error rates and types
- User satisfaction scores
- Model performance metrics

### Alerts
- High error rates
- Slow response times
- Unusual usage patterns
- Model performance degradation

## Security & Safety

### Content Safety
- Input validation and sanitization
- Output filtering for harmful content
- Rate limiting and abuse detection
- PII detection and redaction

### Infrastructure Security
- API key rotation
- Network security groups
- Encrypted data storage
- Audit logging

## Scaling Considerations

### Horizontal Scaling
- Load balancing across API instances
- Vector store sharding
- Async processing queues
- CDN for static assets

### Performance Optimization
- Response caching
- Batch processing
- Model quantization
- Hardware acceleration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License
MIT License - see LICENSE file for details.

## Support
- Documentation: [docs/](docs/)
- Issues: GitHub Issues
- Discussions: GitHub Discussions