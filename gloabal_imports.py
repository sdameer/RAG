import os
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

# loaders 
from langchain_community.document_loaders import TextLoader , PyPDFLoader

# splitters 
from langchain_text_splitters import CharacterTextSplitter , RecursiveCharacterTextSplitter

# Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

# vector stores 
from langchain_community.vectorstores import Chroma

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

# prompts 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# messages 
from langchain_core.messages import HumanMessage, SystemMessage , AIMessage

# llms
from langchain_groq import ChatGroq

# outputs 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda , RunnableSequence

# for pipeline building
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

# vector types
from qdrant_client.models import Distance, VectorParams


client = QdrantClient(url="http://localhost:6333")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


