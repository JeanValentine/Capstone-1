import argparse, json, os, re
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except Exception:
    GRADIO_AVAILABLE = False

STOP = set(stopwords.words("english"))
LEM = WordNetLemmatizer()
SIA = SentimentIntensityAnalyzer()

def load_csv(path: str, nrows: int = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    encodings = ["utf-8", "latin1", "ISO-8859-1"]
    last = None
    for e in encodings:
        try:
            df = pd.read_csv(path, encoding=e, low_memory=False, nrows=nrows)
            print(f"Loaded {path} with encoding {e} rows={len(df)}")
            return df
        except Exception as ex:
            last = ex
    raise last

def clean_text(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^A-Za-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess_text_for_model(s: str) -> str:
    s = clean_text(s).lower()
    toks = [t for t in s.split() if t not in STOP and len(t) > 1]
    toks = [LEM.lemmatize(t) for t in toks]
    return " ".join(toks)

def create_proxy_fake_label(df: pd.DataFrame, helpful_threshold:int=5, short_word_thresh:int=5) -> pd.Series:
    numHelpful_col = next((c for c in ['reviews.numHelpful','numHelpful'] if c in df.columns), None)
    text_col = next((c for c in ['reviews.text','review_text','reviewBody','review'] if c in df.columns), None)
    date_col = next((c for c in ['reviews.date','review_date','date'] if c in df.columns), None)
    username_col = next((c for c in ['reviews.username','username','reviewerName'] if c in df.columns), None)

    fake = pd.Series(0, index=df.index)

    if numHelpful_col:
        fake = fake.where(~(df[numHelpful_col] == 0), 1)
    if text_col:
        lengths = df[text_col].fillna("").astype(str).apply(lambda s: len(s.split()))
        short_mask = lengths <= short_word_thresh
        fake = fake.where(~short_mask, 1)
    if text_col and numHelpful_col:
        pols = df[text_col].fillna("").astype(str).apply(lambda s: SIA.polarity_scores(s)['compound'])
        extreme = (pols >= 0.9) | (pols <= -0.9)
        low_help = df[numHelpful_col].fillna(0).astype(int) <= max(1, helpful_threshold//2)
        fake = fake.where(~(extreme & low_help), 1)
    if username_col and date_col:
        grouped = df.groupby([username_col, date_col]).size()
        suspicious_pairs = grouped[grouped > 3].reset_index()[[username_col, date_col]]
        if not suspicious_pairs.empty:
            dup_mask = df.set_index([username_col, date_col]).index.isin([tuple(x) for x in suspicious_pairs.values])
            fake = fake.where(~dup_mask, 1)

    return fake

def do_eda(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Running EDA...")
    report = {}

    rating_col = next((c for c in ['reviews.rating','rating','stars','reviews.rating.']), None)
    if rating_col and rating_col in df.columns:
        report['rating_stats'] = df[rating_col].describe().to_dict()
    if 'reviews.numHelpful' in df.columns or 'numHelpful' in df.columns:
        col = 'reviews.numHelpful' if 'reviews.numHelpful' in df.columns else 'numHelpful'
        report['helpful_stats'] = df[col].describe().to_dict()
    if 'brand' in df.columns:
        report['top_brands'] = df['brand'].value_counts().head(20).to_dict()
        plt.figure(figsize=(8,6))
        df['brand'].value_counts().head(20).plot.barh()
        plt.title('Top 20 brands by review count'); plt.tight_layout()
        plt.savefig(out_dir / 'top_brands.png'); plt.close()
    if rating_col and rating_col in df.columns:
        plt.figure(figsize=(6,4)); sns.histplot(df[rating_col].dropna(), bins=5); plt.title('Rating distribution'); plt.tight_layout()
        plt.savefig(out_dir / 'rating_distribution.png'); plt.close()
    for cand in ['reviews.date','review_date','date']:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors='coerce')
            ts = df.groupby(pd.Grouper(key=cand, freq='M')).size()
            plt.figure(figsize=(10,3)); ts.plot(); plt.title('Reviews over time (monthly)'); plt.tight_layout()
            plt.savefig(out_dir / 'reviews_time_series.png'); plt.close()
            break
    with open(out_dir / 'eda_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print("EDA artifacts saved to", out_dir)

def train_fake_detector(df: pd.DataFrame, out_dir: Path, sample:int=None, use_smote:bool=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    text_col = next((c for c in ['reviews.text','review_text','reviewBody','review'] if c in df.columns), None)
    if text_col is None:
        raise RuntimeError("No review text column found in dataset.")
    df = df.copy()
    df['fake_label'] = create_proxy_fake_label(df)
    print("Fake label distribution:\n", df['fake_label'].value_counts().to_dict())
    if sample and sample < len(df):
        df = df.sample(sample, random_state=42).reset_index(drop=True)
        print("Using sample of", sample)
    df['clean_text'] = df[text_col].astype(str).apply(preprocess_text_for_model)
    df['review_length'] = df[text_col].astype(str).apply(lambda s: len(str(s).split()))
    num_cols = ['review_length']
    if 'reviews.numHelpful' in df.columns:
        df['numHelpful'] = pd.to_numeric(df['reviews.numHelpful'], errors='coerce').fillna(0)
        num_cols.append('numHelpful')
    X_text = df['clean_text']
    y = df['fake_label'].astype(int)
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(X_text)
    from scipy.sparse import hstack
    X = hstack([X_tfidf, df[num_cols].fillna(0).values])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    if use_smote:
        try:
            sm = SMOTE(random_state=42, n_jobs=-1)
            X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
            X_train, y_train = sm.fit_resample(X_train_dense, y_train)
            print("SMOTE applied; new distribution:", pd.Series(y_train).value_counts().to_dict())
        except Exception as e:
            print("SMOTE failed/skipped:", e)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    }
    results = {}
    for name, model in models.items():
        print("Training", name)
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            report = classification_report(y_test, preds, output_dict=True)
            print(classification_report(y_test, preds))
            joblib.dump(model, out_dir / f"model_{name}.pkl")
            results[name] = report
        except Exception as e:
            print(f"Training failed for {name}:", e)
    joblib.dump(tfidf, out_dir / "tfidf_fake.pkl")
    with open(out_dir / "fake_detector_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Fake detection artifacts saved to", out_dir)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, help="Path to reviews CSV")
    p.add_argument("--out", type=str, default=str(Path.cwd() / "capstone3_out"), help="Output folder")
    p.add_argument("--sample", type=int, default=0, help="Sample N rows for quick runs (0=all)")
    p.add_argument("--do-eda", action="store_true", help="Run EDA")
    p.add_argument("--train-fake", action="store_true", help="Train fake review detector")
    p.add_argument("--use-smote", action="store_true", help="Apply SMOTE for fake detection")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.data or not os.path.exists(args.data):
        raise FileNotFoundError(f"CSV file not found: {args.data}")

    nrows = args.sample if args.sample and args.sample>0 else None
    df = load_csv(args.data, nrows=nrows)
    print(f"Data loaded! Shape: {df.shape}, Columns: {df.columns.tolist()}")

    if args.do_eda:
        do_eda(df.copy(), out_dir)
    if args.train_fake:
        train_fake_detector(df.copy(), out_dir, sample=(args.sample if args.sample>0 else None), use_smote=args.use_smote)

if __name__ == "__main__":
    main()
