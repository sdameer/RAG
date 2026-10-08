import os
import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from _3_final_pipeline import rag_chain, embeddings

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


load_dotenv()


# --------------------------------------------------
# 1. Load evaluation dataset
# --------------------------------------------------

with open("ragas_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)


# --------------------------------------------------
# 2. Run YOUR RAG pipeline on all questions
# --------------------------------------------------

results = []

for item in dataset:

    result = rag_chain.invoke({
        "input": item["user_input"],
        "chat_history": []
    })

    results.append({
        "user_input": item["user_input"],
        "response": result["answer"],
        "retrieved_contexts": [
            doc.page_content
            for doc in result["context"]
        ],
        "reference": item["reference"],
        "reference_contexts": item["reference_contexts"],
    })


# --------------------------------------------------
# 3. Create RAGAS dataset
# --------------------------------------------------

evaluation_dataset = EvaluationDataset.from_list(results)


# --------------------------------------------------
# 4. Groq = RAGAS evaluator LLM
# --------------------------------------------------

groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

evaluator_llm = LangchainLLMWrapper(groq_llm)


# --------------------------------------------------
# 5. HuggingFace = RAGAS evaluator embeddings
# --------------------------------------------------

evaluator_embeddings = LangchainEmbeddingsWrapper(
    embeddings
)


# --------------------------------------------------
# 6. RAGAS metrics
# --------------------------------------------------

metrics = [
    Faithfulness(
        llm=evaluator_llm
    ),

    AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    ),

    ContextPrecision(
        llm=evaluator_llm
    ),

    ContextRecall(
        llm=evaluator_llm
    ),
]


# --------------------------------------------------
# 7. Run RAGAS evaluation
# --------------------------------------------------

result = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics,
)


# --------------------------------------------------
# 8. Print results
# --------------------------------------------------

print("\n========== RAGAS RESULTS ==========\n")

print(result)

print("\n========== SCORES ==========\n")

for metric, score in result.items():
    print(f"{metric}: {score}")