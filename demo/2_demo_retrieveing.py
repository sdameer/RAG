"""
    Retriever the data from the vector store    
"""

import os
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
collection_name = "new_rag_collection"

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": hf_token}
)

db = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,

)

query = "CACHING"

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    