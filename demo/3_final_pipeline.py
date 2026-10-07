"""
    This is a simple perfectly working RAG pipeline
"""

import os
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv


from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
collection_name = "new_rag_collection"

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

load_dotenv()


GROQ_API_KEY = os.environ["GROQ_API_KEY"]

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


path = "C:\\Users\\Syed Ameer Baji\\Desktop\\test\\rags\\demo\\chroma_db"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": hf_token}
)
db = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,

)
retriever = db.as_retriever(search_type="similarity_score_threshold",search_kwargs={"k": 1,"score_threshold":0.8},)
llm = ChatGroq(model="openai/gpt-oss-20b")



history_aware_retriever =   create_history_aware_retriever(
    retriever = retriever,  # get the data from the connected vector db
    prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", (
                        "Given a chat history and the latest user question "
                        "which might reference context in the chat history, "
                        "formulate a standalone question which can be understood "
                        "without the chat history. Do NOT answer the question, just "
                        "reformulate it if needed and otherwise return it as is."
                    )
                    ),  
                    MessagesPlaceholder("chat_history"), # here we will feed the chat history of the user
                    ("human", "{input}"),
                ]
            ), # add that data to the prompt 
    llm = llm # then pass it to the llm 
)




question_answer_chain =  create_stuff_documents_chain(
    # first create a prompt then add the data to the llm 
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", (
                    "You are an assistant for question-answering tasks. Use "
                    "the following pieces of retrieved context to answer the "
                    "question. If you don't know the answer, just say that you "
                    "don't know. Use three sentences maximum and keep the answer "
                    "concise."
                    "\n\n"
                    "{context}"
                    )                  
                ),
                MessagesPlaceholder("chat_history"), # here we will feed the chat history of the user
                ("human", "{input}"),
            ]
    ),
    llm = llm
    )


rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)



def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        
        print(f"AI: {result['answer']}")        
        print("\n--- Sources ---")

        for i, doc in enumerate(result["context"], 1):
            print(f"Source {i}:")
            print(f"File name   : {doc.metadata.get('file_name')}")
            print(f"Page number : {doc.metadata.get('page_number')}")
            print(f"Chunk ID    : {doc.metadata.get('chunk_id')}")
            print("\n")
        
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))



if __name__ == "__main__":
    continual_chat()
