"""
    Retriever the data from the vector store    
"""

import os
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore

load_dotenv()
api_key = os.environ["GOOGLE_API_KEY"] 

# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",output_dimensionality=768, api_key= api_key)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=r"C:\\Users\\Syed Ameer Baji\\Desktop\\test\\rags\\demo\\chroma_db",
    embedding_function=embeddings
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
    