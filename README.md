# NexusIQ

NexusIQ is an intelligent document processing pipeline designed to classify, extract, and recommend actions from unstructured data using RAG (Retrieval-Augmented Generation) and Gemini's LLM capabilities.

## 🚀 Features

- **Document Ingestion**: Advanced parsing of PDFs, Word docs, and emails using Docling.
- **Classification**: Intelligent routing of documents based on content analysis.
- **RAG-based Extraction**: Precise, context-aware data extraction using ChromaDB for local persistence.
- **Validation**: Anomaly detection and business rule validation for extracted data.
- **Recommendation Engine**: Deriving next-best-actions based on extracted insights.
- **Evaluation**: Ragas-based metrics for measuring retrieval and generation quality.

## 📂 Project Structure

```text
├── data/                          # Data persistence and raw files
├── db/                            # Database files
├── prompts/                       # Externalized prompt templates
├── src/                           # Main application logic
│   ├── ingestion/                 # Document parsing logic
│   ├── llm/                       # Gemini client and prompt management
│   ├── rag/                       # ChromaDB and retrieval logic
│   ├── core/                      # Main business logic (classifier, extractor, etc.)
│   ├── pipeline/                  # Pipeline orchestration
│   ├── evaluation/                # Metrics and performance monitoring
│   └── utils/                     # Formatting and logging
├── tests/                         # Unit and integration tests
├── cli.py                         # entry point of the project
├── config.py                      # important config params that need to setup before hand
└── requirements.txt               # Project dependencies
```

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- Gemini API Key

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/anikesh-mane/nexusiq.git
    cd nexusiq
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure environment variables**:
    Copy `.env.example` to `.env` and fill in your API keys.
    ```bash
    cp .env.example .env
    ```
    update the env and config.py files

## 🖥️ Usage

Run the CLI for document processing:

```bash

# chatbot runs by default in these cases
python src/cli.py data/raw/sample.pdf

python -m src.cli data/raw/sample.pdf

# Default — processes document, then opens chat session
python -m src.cli data/raw/invoice.pdf

# Skip the chatbot (e.g. in CI/scripts)
python -m src.cli data/raw/invoice.pdf --no-chat

# Save JSON output AND chat
python -m src.cli data/raw/invoice.pdf -o output.json

```

## 🧪 Testing

Run tests using pytest:

```bash
pytest tests/
```

## 📝 License

MIT
