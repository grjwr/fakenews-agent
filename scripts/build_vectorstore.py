import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os, time

df = pd.read_csv("data/processed/train.csv")
print(f"Building index from {len(df)} documents...")

docs = [
    Document(
        page_content=row['title'].strip(),
        metadata={"label": row['label'], "source": row['source']}
    )
    for _, row in df.iterrows()
]

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 64},
)

BATCH = 5000
print(f"Embedding {len(docs)} docs in batches of {BATCH}...")
t0 = time.time()

vectorstore = None
for i in range(0, len(docs), BATCH):
    batch = docs[i:i+BATCH]
    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch, embeddings)
    else:
        vectorstore.add_documents(batch)
    print(f"  [{min(i+BATCH, len(docs))}/{len(docs)}] done — {time.time()-t0:.0f}s elapsed")

os.makedirs("data/fakenews_index", exist_ok=True)
vectorstore.save_local("data/fakenews_index")
print(f"\nDone. Total time: {time.time()-t0:.0f}s")
