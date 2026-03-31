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
|   ├── ingestion/                 # Document parsing logic
|   ├── llm/                       # Gemini client and prompt management
|   ├── rag/                       # ChromaDB and retrieval logic
|   ├── core/                      # Main business logic (classifier, extractor, etc.)
|   ├── pipeline/                  # Pipeline orchestration
|    ├── evaluation/                # Metrics and performance monitoring
|    └── utils/                     # Formatting and logging
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

## Sample outputs

![Immediate Processing Output](data/artifacts/image.png)

```bash
{                                                                                                                                                                                                                            
   "document": "invoice_anomaly.txt",                                                                                                                                                                                         
   "document_type": "invoice",                                                                                                                                                                                                
   "confidence_score": 1.0,                                                                                                                                                                                                   
   "classification_reasoning": "The document explicitly states 'Invoice' and contains typical invoice fields such as vendor name, invoice ID, dates, bill-to information, itemized list with quantities and prices, total amo 
   "key_entities": {                                                                                                                                                                                                          
     "vendor": "Beta Industrial Supplies",                                                                                                                                                                                    
     "buyer": "Zenith Manufacturing Ltd.",                                                                                                                                                                                    
     "invoice_number": "BIS-7781",                                                                                                                                                                                            
     "amount": 425000.0,                                                                                                                                                                                                      
     "currency": "INR",                                                                                                                                                                                                       
     "date": "25 Feb 2026",                                                                                                                                                                                                   
     "due_date": "20 Feb 2026",                                                                                                                                                                                               
     "line_items": [                                                                                                                                                                                                          
       {                                                                                                                                                                                                                      
         "description": "Hydraulic Pump Model XZ",                                                                                                                                                                            
         "quantity": 5.0,                                                                                                                                                                                                     
         "unit_price": 85000.0,                                                                                                                                                                                               
         "total": 425000.0                                                                                                                                                                                                    
       }                                                                                                                                                                                                                      
     ],                                                                                                                                                                                                                       
     "payment_terms": "Net 30 days"                                                                                                                                                                                           
   },                                                                                                                                                                                                                         
   "validation": {                                                                                                                                                                                                            
     "issue_count": 1,                                                                                                                                                                                                        
     "issues": [                                                                                                                                                                                                              
       {                                                                                                                                                                                                                      
         "type": "suspicious_value",                                                                                                                                                                                          
         "message": "Invoice amount (425,000.00) exceeds threshold (100,000). Flag for review.",                                                                                                                              
         "severity": "medium"                                                                                                                                                                                                 
       }                                                                                                                                                                                                                      
     ]                                                                                                                                                                                                                        
   },                                                                                                                                                                                                                         
   "recommendations": [                                                                                                                                                                                                       
     {                                                                                                                                                                                                                        
       "action": "Review the invoice amount for accuracy and legitimacy.",                                                                                                                                                    
       "reasoning": "The invoice amount of 425,000.00 INR significantly exceeds the typical threshold, warranting a detailed review to prevent potential financial discrepancies or fraud.",                                  
       "priority": "High"                                                                                                                                                                                                     
     },                                                                                                                                                                                                                       
     {                                                                                                                                                                                                                        
       "action": "Verify the invoice date and due date for logical consistency.",                                                                                                                                             
       "reasoning": "The due date (20 Feb 2026) appears to be before the invoice date (25 Feb 2026), which is highly unusual and could indicate a data entry error or an incorrect payment schedule.",                        
       "priority": "High"                                                                                                                                                                                                     
     },                                                                                                                                                                                                                       
     {                                                                                                                                                                                                                        
       "action": "Cross-reference the invoice with the corresponding Purchase Order (PO).",                                                                                                                                   
       "reasoning": "Given the high amount and potential date discrepancy, comparing the invoice details (items, quantities, prices, and total) against the original PO will confirm the validity of the charges and ensure a 
       "priority": "Medium"                                                                                                                                                                                                   
     }                                                                                                                                                                                                                                                                          
   ],                                                                                                                                                                                                                         
   "processing_time_seconds": 16.397                                                                                                                                                                                          
 }                                                                 
 ```                                                                                                                                                        
![Chatbot Output](data/artifacts/image_copy.png)

## 🧪 Testing

Run tests using pytest:

```bash
pytest tests/
```

## 📝 License

MIT
