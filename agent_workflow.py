import re
from typing import List
from vector_store import load_vector_store, retrieve_top_chunks
from llm_integration import generate_answer
from sentence_transformers import SentenceTransformer
import os
import requests

def get_word_definition(word: str) -> str:
    """Fetch the definition of a word using a dictionary API."""
    api_url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list):
                meanings = data[0].get("meanings", [])
                if meanings:
                    definitions = meanings[0].get("definitions", [])
                    if definitions:
                        return definitions[0].get("definition", "No definition found.")
        return "Definition not found."
    except Exception as e:
        return f"Error fetching definition: {e}"

def agent_workflow(query: str, vector_store, model, documents: List[str]) -> str:
    """Agent workflow to handle queries and route to appropriate tools."""
    # Log the query
    print(f"Received query: {query}")

    # Check for keywords to route to specific tools
    if any(keyword in query.lower() for keyword in ["calculate", "compute"]):
        # Convert query to lowercase for consistent splitting
        lower_query = query.lower()
        print("Debug: Splitting query for calculation:", lower_query.split("calculate", 1) if "calculate" in lower_query else lower_query.split("compute", 1))
        # Extract the expression to calculate
        parts = lower_query.split("calculate", 1) if "calculate" in lower_query else lower_query.split("compute", 1)
        expression = parts[1].strip() if len(parts) > 1 else ""
        if expression:
            try:
                result = eval(expression)
                return f"Calculation result: {result}"
            except Exception as e:
                return f"Error in calculation: {e}"
        else:
            return "Error: No expression provided to calculate."

    if "define" in query.lower():
        # Convert query to lowercase for consistent splitting
        lower_query = query.lower()
        print("Debug: Splitting query for 'define':", lower_query.split("define", 1))
        # Extract the word to define
        parts = lower_query.split("define", 1)
        word = parts[1].strip() if len(parts) > 1 else ""
        if word:
            definition = get_word_definition(word)
            return f"Definition of {word}: {definition}"
        else:
            return "Error: No word provided to define."

    # Default to RAG pipeline
    print("Executing RAG pipeline...")
    context = retrieve_top_chunks(query, vector_store, model, documents)
    answer = generate_answer(context, query)

    # Return both context and answer separately
    return context, answer

if __name__ == "__main__":
    # Load vector store and documents
    index_path = "./data/vector_index"
    chunks_folder = "./data/chunks"

    print("Loading vector store...")
    vector_store = load_vector_store(index_path)

    print("Loading documents...")
    documents = []
    for file_name in os.listdir(chunks_folder):
        file_path = os.path.join(chunks_folder, file_name)
        if file_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())

    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Example queries
    queries = [
        "What is Stripe's mission?",  # Updated query to reflect Stripe content
        "Calculate 10 * 5",
        "Define economic infrastructure"
    ]

    for query in queries:
        print("\nProcessing query:", query)
        response = agent_workflow(query, vector_store, model, documents)
        print("Response:", response)
