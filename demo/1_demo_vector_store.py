"""
    Load the data 
    and convert the data to vector store
"""

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import   PyPDFLoader
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

    
data_dir = r"C:\\Users\\Syed Ameer Baji\\Desktop\\test\\rags\\demo\\data"

# create a list of all the pdf documents 
pdf_list = []
for file in os.listdir(data_dir):
    if file.lower().endswith(".pdf"):
        pdf_list.append(file)
    
# extracting only the first 10 pages of each of the pdf document present in the pdf_list
# new path = dir path + file path 
# load the pdf
# extract 10 pages
# add metadata for every page
documents = []
for pdf_file in pdf_list:
    print(f"file : {pdf_file}")
    new_path = os.path.join(data_dir , pdf_file)
    loader = PyPDFLoader(new_path)
    pages = loader.load()[:10]
    for page in pages :
        page.metadata['file_name'] = pdf_file
        page.metadata['page_number'] = page.metadata.get("page", 0) + 1
        documents.append(page)
    
print(documents)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
documents = text_splitter.split_documents(documents=documents)

for chunk_id , document in enumerate(documents):
    print(f"{chunk_id} - {document}\n")
    document.metadata['chunk_id'] = chunk_id

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



print("Vector store created successfully.")
print(f"PDF files: {len(pdf_list)}")
print(f"Total chunks: {len(documents)}")