import os
from typing import List
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(folder_path: str) -> List[str]:
    """Load all text documents from a folder."""
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

def chunk_documents(documents: List[str], chunk_size: int = 300, chunk_overlap: int = 50) -> List[str]:
    """Chunk documents into smaller pieces."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))
    return chunks

def save_chunks(chunks: List[str], output_folder: str):
    """Save chunks to the output folder."""
    os.makedirs(output_folder, exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(os.path.join(output_folder, f'chunk_{i}.txt'), 'w', encoding='utf-8') as file:
            file.write(chunk)

if __name__ == "__main__":
    # Load and chunk documents
    input_folder = "./documents"  # Updated to use Stripe documents
    output_folder = "./data/chunks"

    print("Loading documents...")
    docs = load_documents(input_folder)

    print("Chunking documents...")
    chunks = chunk_documents(docs)

    print(f"Saving {len(chunks)} chunks...")
    save_chunks(chunks, output_folder)

    print("Data ingestion completed.")
