import os
import pinecone
from pinecone import Pinecone
import tiktoken
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Get Pinecone API key & index name from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_HOST_NAME = os.getenv("PINECONE_HOST_NAME")

# Load the open-source embedding model (1536-D)
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Initialize Pinecone client
# pc = Pinecone(api_key=PINECONE_API_KEY, host=PINECONE_HOST_NAME)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to an existing index
index = pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST_NAME)

# Tokenizer for chunking
tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, chunk_size=512, overlap=100):
    """Splits text into overlapping chunks for better retrieval accuracy."""
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))

    return chunks

def store_embeddings(session_id, text):
    """Store text chunks directly in Pinecone using built-in embeddings."""
    chunks = chunk_text(text)
    vectors = embedding_model.encode(chunks).tolist()
    
    for i, chunk in enumerate(chunks):
        # Pinecone automatically generates embeddings, so we only store the text
        pinecone_data = [
            (f"{session_id}_{i}", vectors[i], {"session_id": session_id, "text": chunk})
        ]
        index.upsert(vectors=pinecone_data)

def retrieve_embeddings(session_id, query, top_k=5):
    """Retrieve top relevant chunks using Pinecone’s built-in embeddings."""
    query_vector = embedding_model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True, query_text=query)

    # Filter results to return only text chunks for this session
    relevant_chunks = [
        match["metadata"]["text"] for match in results["matches"]
        if match["metadata"]["session_id"] == session_id
    ]

    return relevant_chunks

def delete_session_embeddings(session_id):
    """Delete all embeddings related to a session when it expires."""
    try:
        for ids in index.list(prefix=session_id):
            index.delete(ids=ids)
        print(f"Successfully deleted vectors for session: {session_id}")
        return True
            
    except Exception as e:
        print(f"Error in delete_session_embeddings: {str(e)}")
        raise