"""
    Load the data 
    and convert the data to vector store
"""

import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore

load_dotenv()
api_key = os.environ["GOOGLE_API_KEY"] 

# get data from here 
file_path = r"C:\\Users\\Syed Ameer Baji\\Desktop\\test\\rags\\rag_main_files\\data.txt"

# TextLoader converts a file into LangChain Document objects.
loader = TextLoader(file_path)
documents = loader.load()

# break down the data ino 200 parts 
# with each parts last 10 words repeating 
text_splitter = CharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 10
)

# convert to vectors
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",output_dimensionality=768, api_key= api_key)

# create vector store
db = Chroma.from_documents(
    documents, 
    embedding=embeddings,
    persist_directory= "C:\\Users\\Syed Ameer Baji\\Desktop\\test\\rags\\rag_main_files\\data.txt"
)

