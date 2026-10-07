"""
    Load the data 
    and convert the data to vector store
"""

import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")
collection_name = "new_rag_collection"

# create collection if it does not exists
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    
file_path = r"C:\\Users\\Syed Ameer Baji\\Desktop\\test\\rags\\demo\\data\\data.txt"
loader = TextLoader(file_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
documents = text_splitter.split_documents(documents=documents)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": hf_token}
)

# create vector store
print("-----\tCreating Vector Store\t-----")
db = QdrantVectorStore.from_documents(
    documents,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name=collection_name,
)

print("-----\tVector Store Created\t-----")
