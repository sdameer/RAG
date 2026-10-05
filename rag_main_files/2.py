"""
    Retriever the data from the vector store    
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore

load_dotenv()
api_key = os.environ["GOOGLE_API_KEY"] 

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",output_dimensionality=768, api_key= api_key)


db = Chroma(
    persist_directory=r"C:\\Users\\Syed Ameer Baji\\Desktop\\test\\rags\\1\\chroma_db",
    embedding_function=embeddings
)

query = "Retrieval-Augmented Generation (RAG)"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5},
)

relevent_docs = retriever.invoke(query)

print(relevent_docs.)