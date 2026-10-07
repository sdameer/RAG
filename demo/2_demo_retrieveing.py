"""
    Retriever the data from the vector store    
"""

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings('ignore')


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

query = "Perceptron training is learning by imitation, which is called ‘supervised learn-ing’."

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"file name   : {doc.metadata.get('file_name', None)}")
        print(f"page number : {doc.metadata.get('page_number', None)}")
        #  get all the metadata
        # for key , value in doc.metadata.items():
        #     print(f"- {key} :\t{value}")



