from langchain_core.prompts import ChatPromptTemplate

# # PART 1:
template = "Tell me {n} jokes about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({"n": 3, "topic": "cats"})
print(prompt)

# PART 2 :
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes.")
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"n": "3", "x": 3})
print(prompt)
