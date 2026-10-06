from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda , RunnableSequence

model = ChatGroq(model="openai/gpt-oss-20b")


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# method - 1
chain = prompt_template | model | StrOutputParser()

# method - 2
chain = prompt_template | model | StrOutputParser() | RunnableLambda(lambda x: x.upper())

# method - 3 
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_message()))  # type: ignore
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)



result = chain.invoke({"topic": "lawyers", "joke_count": 3})