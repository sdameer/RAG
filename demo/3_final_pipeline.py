import os
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
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

api_key = os.environ["GOOGLE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


path = "C:\\Users\\Syed Ameer Baji\\Desktop\\test\\rags\\demo\\chroma_db"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": hf_token}
)
db = Chroma(persist_directory=path, embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity",search_kwargs={"k": 3},)
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
        
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))



if __name__ == "__main__":
    continual_chat()
