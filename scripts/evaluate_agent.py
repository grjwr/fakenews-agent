import sys, json
sys.path.insert(0, ".")
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from agent.graph import graph
from agent.state import AgentState

DATASETS = ["politifact", "gossipcop", "welfake", "buzzfeed", "liar2", "fake_real"]

print(f"\n{'Dataset':12s} | {'Samples':7} | {'Accuracy':8} | {'F1':8} | {'Precision':9} | {'Recall':8}")
print("-"*70)

all_results = []
for name in DATASETS:
    df = pd.read_csv("data/processed/test.csv")
    df = df[df["source"] == name].head(100)  # 100 per dataset
    if len(df) == 0:
        continue

    y_true, y_pred = [], []
    for _, row in df.iterrows():
        state: AgentState = {
            "claim": row["title"], "messages": [], "evidence": [],
            "eprvfl_result": None, "mistral_result": None,
            "verdict": None, "route": "fast",
        }
        try:
            result = graph.invoke(state)
            pred = result["verdict"]["label"]
        except:
            pred = "FAKE"
        y_true.append(row["label"])
        y_pred.append(pred)

    m = {
        "dataset":   name,
        "samples":   len(df),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred, pos_label="FAKE", average="binary", zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, pos_label="FAKE", average="binary", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, pos_label="FAKE", average="binary", zero_division=0), 4),
    }
    all_results.append(m)
    print(f"{name:12s} | {len(df):7d} | {m['accuracy']:.4f}   | {m['f1']:.4f}   | {m['precision']:.4f}    | {m['recall']:.4f}")

import os
os.makedirs("results", exist_ok=True)
with open("results/agent_eval.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to results/agent_eval.json")
