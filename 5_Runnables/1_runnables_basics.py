from langchain_core.runnables import RunnableLambda


def clean_query(query):
    query = query.strip()
    query = query.lower()
    query = query.replace("!", "")
    return query


cleaner = RunnableLambda(clean_query)

result = cleaner.invoke("   My payment has FAILED!!!   ")

print(result)

from langchain_core.runnables import RunnableParallel, RunnablePassthrough


def get_documents(question):
    return "Refunds are processed within 7 working days."


rag_input = RunnableParallel({
    "context": get_documents,
    "question": RunnablePassthrough()
})

result = rag_input.invoke("What is the refund policy?")

print(result)