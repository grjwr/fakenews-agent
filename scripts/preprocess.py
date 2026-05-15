import pandas as pd
import os

# Label mapping: 1=FAKE, 0=REAL (consistent across all your datasets)
def load_dataset(path, name, encoding='utf-8'):
    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin-1')  # fallback for BuzzFeed
    
    df = df[['title', 'verdict']].dropna()
    df['verdict'] = df['verdict'].astype(int)
    df['label'] = df['verdict'].map({1: 'FAKE', 0: 'REAL'})
    df['source'] = name
    df = df[['title', 'label', 'source']]
    print(f"{name:12s} | {len(df):6d} rows | FAKE: {(df.label=='FAKE').sum():6d} | REAL: {(df.label=='REAL').sum():6d}")
    return df

datasets = {
    "politifact": "/home/akumar/llm_fakenews/politifact_final.csv",
    "gossipcop":  "/home/akumar/llm_fakenews/GossipCop_final.csv",
    "welfake":    "/home/akumar/llm_fakenews/WELFake_final.csv",
    "buzzfeed":   "/home/akumar/llm_fakenews/BuzzFeed.csv",
    "fake_real":  "/home/akumar/llm_fakenews/fake_and_real_news_dataset.csv",
    "liar2":      "/home/akumar/llm_fakenews/LIAR2_skewed_final.csv",
}

print("="*60)
print(f"{'Dataset':12s} | {'Rows':>6} | {'FAKE':>8} | {'REAL':>8}")
print("="*60)

dfs = []
for name, path in datasets.items():
    df = load_dataset(path, name)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined = combined.drop_duplicates(subset=['title'])
combined = combined[combined['title'].str.len() > 10]  # remove junk rows

print("="*60)
print(f"{'TOTAL':12s} | {len(combined):6d} rows | FAKE: {(combined.label=='FAKE').sum():6d} | REAL: {(combined.label=='REAL').sum():6d}")
print("="*60)

# Save splits
os.makedirs("data/processed", exist_ok=True)
combined.to_csv("data/processed/combined_all.csv", index=False)

# 80/10/10 split per dataset (stratified)
from sklearn.model_selection import train_test_split

train_parts, val_parts, test_parts = [], [], []
for name in combined['source'].unique():
    subset = combined[combined['source'] == name]
    tr, temp = train_test_split(subset, test_size=0.2, random_state=42, stratify=subset['label'])
    va, te  = train_test_split(temp,   test_size=0.5, random_state=42, stratify=temp['label'])
    train_parts.append(tr)
    val_parts.append(va)
    test_parts.append(te)

train = pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
val   = pd.concat(val_parts).sample(frac=1, random_state=42).reset_index(drop=True)
test  = pd.concat(test_parts).sample(frac=1, random_state=42).reset_index(drop=True)

train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv",   index=False)
test.to_csv("data/processed/test.csv",  index=False)

print(f"\nSplits saved:")
print(f"  train: {len(train):6d} rows -> data/processed/train.csv")
print(f"  val:   {len(val):6d} rows -> data/processed/val.csv")
print(f"  test:  {len(test):6d} rows -> data/processed/test.csv")
