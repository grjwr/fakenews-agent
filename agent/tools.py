import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 64},
        )
        _vectorstore = FAISS.load_local(
            "data/fakenews_index", emb,
            allow_dangerous_deserialization=True
        )
        print("[RAG] Vector store loaded.")
    return _vectorstore
