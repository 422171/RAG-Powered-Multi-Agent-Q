import streamlit as st
from agent_workflow import agent_workflow
from vector_store import load_vector_store, retrieve_top_chunks
from sentence_transformers import SentenceTransformer
import os

# Load vector store and documents
index_path = "./data/vector_index"
chunks_folder = "./data/chunks"

st.title("RAG-Powered Multi-Agent Q&A Assistant")

@st.cache_resource
def load_resources():
    st.write("Loading vector store and model...")
    vector_store = load_vector_store(index_path)
    documents = []
    for file_name in os.listdir(chunks_folder):
        file_path = os.path.join(chunks_folder, file_name)
        if file_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return vector_store, documents, model

vector_store, documents, model = load_resources()

# Input query
query = st.text_input("Enter your query:")

if query:
    st.write(f"Processing query: {query}")
    # Process the query and handle RAG pipeline separately
    if "calculate" in query.lower() or "compute" in query.lower() or "define" in query.lower():
        response = agent_workflow(query, vector_store, model, documents)
        context = None  # No context for non-RAG pipelines
    else:
        context, response = agent_workflow(query, vector_store, model, documents)

    # Display the tool/agent branch used
    st.sidebar.write("### Tool/Agent Branch Used:")
    if "calculate" in query.lower() or "compute" in query.lower():
        st.sidebar.write("Calculator Tool")
    elif "define" in query.lower():
        st.sidebar.write("Dictionary Tool")
    else:
        st.sidebar.write("RAG Pipeline")

    # Display retrieved context snippets in the aside
    if context:
        st.sidebar.write("### Retrieved Context Snippets:")
        for snippet in context:
            st.sidebar.write(f"- {snippet}")

    # Display only the answer in the main section
    st.write("### Response:")
    st.write(response)
