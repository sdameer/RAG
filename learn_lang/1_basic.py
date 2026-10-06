import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

model = ChatGroq(model="openai/gpt-oss-20b")

# simple model usage
res = model.invoke("what is 1+1+2")
print(res.content)

    
# human and system 
msg = [
    SystemMessage(content="you are an excellent math teacher"),
    HumanMessage(content="what is 1 + 1 - 1 + 0 = ")
]
res = model.invoke(msg)
print(res.content)
    

# simple AI assisstant with chat history
history = []
history.append(SystemMessage(content="you are a simple AI math assistant"))
print("Enter exit to stop !!! ")
while True:
    query = input("you : ").lower()
    if query == "exit":
        break
    res = model.invoke(history)
    print("AI : ",res)
    
    history.append(HumanMessage(content = query))
    history.append(SystemMessage(content=res.content))
        
        