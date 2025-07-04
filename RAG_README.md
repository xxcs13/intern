# RAG Q&A System

A simple Retrieval-Augmented Generation (RAG) system that can answer questions based on PDF, PowerPoint, and Excel files using OpenAI's API and LangGraph framework.

## Features

- **Multi-format support**: PDF, PPTX, and XLS files
- **LangGraph workflow**: Structured processing pipeline
- **OpenAI embeddings**: Using text-embedding-3-small model
- **ChromaDB**: Local vector database (no additional costs)
- **Logging**: Automatic logging of Q&A sessions to CSV

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

3. **Prepare your documents**:
   - Place your PDF, PPTX, and XLS files in the project directory
   - Update the file paths in `rag.py` if needed

## Usage

### Basic Usage

Run the RAG system with default settings:
```bash
python rag.py
```

### Testing

Run the test script to verify everything works:
```bash
python test_rag.py
```

### Custom Questions

Modify the `question` field in the `state` dictionary in `rag.py`:
```python
state = {
    "pdf_path": "./your_pdf_file.pdf",
    "pptx_path": "./your_pptx_file.pptx",
    "xls_path": "./your_xls_file.xls",
    "question": "Your custom question here"
}
```

## Workflow

The RAG system follows this pipeline:

1. **Document Parsing**: Extract text from PDF, PPTX, and XLS files
2. **Embedding**: Create vector embeddings using OpenAI's embedding model
3. **Query Processing**: Search for relevant chunks based on the question
4. **Answer Generation**: Use OpenAI's LLM to generate an answer
5. **Logging**: Save the Q&A session to CSV file

## Configuration

- **Chunk size**: Default 1000 characters with 100 character overlap
- **Top-k retrieval**: Default 5 most relevant chunks
- **LLM model**: gpt-3.5-turbo
- **Embedding model**: text-embedding-3-small
- **Temperature**: 0.1 for consistent answers

## Output

- **Console output**: Question and answer displayed
- **CSV log**: All Q&A sessions logged to `rag_qa_log.csv`
- **Vector database**: Stored in `chroma_db/` directory

## Cost Optimization

- Uses ChromaDB for local vector storage (no cloud costs)
- Only OpenAI API calls for embeddings and LLM generation
- Efficient chunking strategy to minimize API calls

## Error Handling

The system includes error handling for:
- Missing files
- API failures
- Document parsing errors
- Vector database issues

## Example Files

The default configuration expects these files:
- `tsmc_2024_yearly report.pdf`
- `無備忘錄版_digitimes_2025年全球AI伺服器出貨將達181萬台　高階機種採購不再集中於四大CSP.pptx`
- `excel_FS-Consolidated_1Q25.xls`

Update the file paths in `rag.py` to match your actual files.
