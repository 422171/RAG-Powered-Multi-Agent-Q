import os
from typing import List
from sentence_transformers import SentenceTransformer

def build_vector_store(chunks_folder: str, index_path: str):
    """Build a FAISS vector store from document chunks."""
    # Load chunks as documents
    documents = []
    for file_name in os.listdir(chunks_folder):
        file_path = os.path.join(chunks_folder, file_name)
        if file_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())

    # Initialize embeddings using SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, convert_to_tensor=True)

    # Initialize FAISS vector store
    import faiss
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.cpu().numpy())

    # Save the vector store
    faiss.write_index(index, index_path)
    print(f"Vector store saved to {index_path}")

def load_vector_store(index_path: str):
    """Load an existing FAISS vector store."""
    import faiss
    return faiss.read_index(index_path)

def retrieve_top_chunks(query: str, vector_store, model, documents: List[str], top_k: int = 3) -> List[str]:
    """Retrieve the top-k relevant chunks for a query."""
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = vector_store.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

if __name__ == "__main__":
    chunks_folder = "./data/chunks"
    index_path = "./data/vector_index"

    print("Building vector store...")
    documents = []
    for file_name in os.listdir(chunks_folder):
        file_path = os.path.join(chunks_folder, file_name)
        if file_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    build_vector_store(chunks_folder, index_path)

    print("Loading vector store...")
    vector_store = load_vector_store(index_path)

    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Test retrieval
    query = "What is Stripe's mission?"  # Updated test query to reflect Stripe content
    print("Retrieving top chunks for query:", query)
    top_chunks = retrieve_top_chunks(query, vector_store, model, documents)
    print("Top chunks:", top_chunks)
