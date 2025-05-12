# RAG-Powered Multi-Agent Q&A Assistant

## Overview
This project implements a Retrieval-Augmented Generation (RAG)-powered multi-agent Q&A assistant. The assistant is designed to:

1. Retrieve relevant information from a document collection using a vector store.
2. Generate natural-language answers using a language model (GPT-2).
3. Orchestrate retrieval and generation steps with an agentic workflow.
4. Provide a user-friendly interface to display the tool/agent branch used, retrieved context snippets, and the complete answer.

## Architecture

### Key Components
1. **Data Ingestion**:
   - Processes documents and chunks them for vector indexing.
   - Stores chunks in the `data/chunks` folder.

2. **Vector Store & Retrieval**:
   - Uses FAISS for efficient vector indexing and retrieval.
   - Retrieves top-k relevant chunks for a given query.

3. **LLM Integration**:
   - Utilizes GPT-2 for text generation.
   - Combines retrieved context and query to generate answers.

4. **Agent Workflow**:
   - Routes queries to appropriate tools based on keywords (e.g., "calculate," "define").
   - Defaults to the RAG pipeline for other queries.

5. **Streamlit Interface**:
   - Provides an interactive UI for users to input queries.
   - Displays the tool/agent branch used, retrieved context snippets, and the generated answer.

### Key Design Choices
- **RAG Pipeline**: Combines retrieval and generation to provide contextually relevant answers.
- **Agentic Workflow**: Routes queries to specialized tools (e.g., calculator, dictionary) for better accuracy.
- **Streamlit UI**: Ensures a user-friendly interface for interaction and debugging.

## How to Run

### Prerequisites
- Python 3.8 or higher
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Steps to Run
1. Start the Streamlit server:
   ```bash
   streamlit run demo_interface.py
   ```
2. Open the app in your browser (default: `http://localhost:8501`).
3. Enter your query in the input box and view the results.

### Example Queries
- "What is Stripe's mission?"
- "Calculate 10 * 5"
- "Define economic infrastructure"

## Folder Structure
- `data/`: Contains vector index and document chunks.
- `documents/`: Original documents used for retrieval.
- `notebooks/`: Jupyter notebooks for experimentation.
- `demo_interface.py`: Streamlit-based user interface.
- `agent_workflow.py`: Implements the agentic workflow.
- `llm_integration.py`: Handles LLM-based text generation.
- `vector_store.py`: Manages vector indexing and retrieval.

## Future Improvements
- Support for additional tools (e.g., translation, summarization).
- Integration with more advanced LLMs (e.g., GPT-3, GPT-4).
- Enhanced UI/UX for better user interaction.
