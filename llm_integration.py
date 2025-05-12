from typing import List
from transformers import pipeline

def generate_answer(context: List[str], query: str) -> str:
    """Generate an answer using an open-source LLM based on the context and query."""
    # Combine context into a single string
    combined_context = "\n".join(context)

    # Initialize a text generation pipeline (using Hugging Face Transformers)
    generator = pipeline("text-generation", model="gpt2")

    # Generate the answer
    prompt = f"{query}\n\n{combined_context}\n\nAnswer:"
    response = generator(prompt, max_new_tokens=300, num_return_sequences=1)

    # Extract only the answer part from the generated text
    answer = response[0]['generated_text'].split("Answer:", 1)[-1].strip()
    return answer

if __name__ == "__main__":
    # Example usage
    context = [
        "This is a sample document for testing the RAG-powered assistant. It contains information about the assistant's capabilities and features.",
        "The assistant can retrieve relevant information from a document collection and generate natural language answers using an LLM.",
        "The assistant uses a vector store for efficient retrieval and supports agentic workflows for decision-making."
    ]
    query = "What are the capabilities of the assistant?"

    print("Generating answer...")
    answer = generate_answer(context, query)
    print("Answer:", answer)
