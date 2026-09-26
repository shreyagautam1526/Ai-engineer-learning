from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "The capital of France is Paris.",
    "The capital of India is New Delhi.",
    "The capital of Japan is Tokyo."
]

embedding_vectors = embeddings.embed_documents(documents)
print(str(embedding_vectors))