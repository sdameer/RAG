"""
    Load the data 
    and convert the data to vector store
"""

import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore


load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
api_key = os.environ["GOOGLE_API_KEY"] 

if not os.path.exists("C:\\Users\\Syed Ameer Baji\\Desktop\\test\\rags\\demo\\chroma_db"):
    
    # get data from here 
    file_path = r"C:\\Users\\Syed Ameer Baji\\Desktop\\test\\rags\\demo\\data\\data.txt"

    # TextLoader converts a file into LangChain Document objects.
    loader = TextLoader(file_path)
    documents = loader.load()

    # break down the data ino 200 parts 
    # with each parts last 10 words repeating 
    text_splitter = CharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 20
    )
    documents = text_splitter.split_documents(documents=documents)

    # convert to vectors
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token": hf_token}
    )

    # create vector store
    print("-----\tCreating Vector Store\t-----")
    db = Chroma.from_documents(
        documents, 
        embedding=embeddings,
        persist_directory= "C:\\Users\\Syed Ameer Baji\\Desktop\\test\\rags\\demo\\chroma_db"
    )
    print("-----\tVector Store Created\t-----")
else : 
    print("-----\tVector Store Exists\t-----")

