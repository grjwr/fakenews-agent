from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
vs = FAISS.load_local(
    "data/fakenews_index", emb,
    allow_dangerous_deserialization=True
)

test_claims = [
    "COVID vaccines cause autism and contain microchips",
    "NASA confirms water found on the moon",
    "Barack Obama was not born in the United States",
]

for claim in test_claims:
    print(f"\nClaim: {claim}")
    results = vs.similarity_search_with_score(claim, k=3)
    for doc, score in results:
        print(f"  [{doc.metadata['label']:4s}] (score={score:.3f}) {doc.page_content[:80]}")
