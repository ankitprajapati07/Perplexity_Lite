import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize FAISS and Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
d = 384  # Dimension of MiniLM embeddings
index = faiss.IndexFlatL2(d)  # FAISS index for similarity search
stored_texts = []  # To keep track of original messages

def chunk_text(text, chunk_size=250):
    """Splits long text into smaller chunks."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def store_message(message):
    """Stores message in FAISS, chunking if necessary."""
    global index, stored_texts
    chunks = chunk_text(message)  # Split long text into chunks
    for chunk in chunks:
        embedding = model.encode(chunk)
        index.add(np.array([embedding], dtype=np.float32))
        stored_texts.append(chunk)  # Store the chunk

def retrieve_relevant_messages(query, top_k=5):
    """Retrieves the most relevant stored chunks based on similarity search."""
    if index.ntotal == 0:
        return []  # No stored messages yet
    query_embedding = model.encode(query)
    D, I = index.search(np.array([query_embedding], dtype=np.float32), k=top_k)
    return [stored_texts[i] for i in I[0] if i < len(stored_texts)]

# Example Usage
# store_message("This is a very long message that should be chunked into smaller parts before storing in FAISS.")
# retrieved = retrieve_relevant_messages("long message")
# print("Relevant Chunks:", retrieved)
