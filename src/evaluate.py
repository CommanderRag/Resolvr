"""
Run the pipeline over the golden evaluation set and compute metrics.

This is the fast, reproducible path (should complete in well under 15 minutes
given the precomputed grounding index in data/). It does NOT rebuild the
thread data or the grounding index from scratch -- see data_prep.py for that
slower, one-time step.

Usage:
    python src/evaluate.py
"""
import argparse
import pickle
import time

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report

from pipeline import run_pipeline, classify_intent_baseline


def load_artifacts(data_dir: str):
    with open(f"{data_dir}/grounding_pairs.pkl", "rb") as f:
        grounding_pairs = pickle.load(f)
    with open(f"{data_dir}/grounding_embeddings.pkl", "rb") as f:
        grounding_embeddings = pickle.load(f)
    with open(f"{data_dir}/baseline_classifier.pkl", "rb") as f:
        baseline = pickle.load(f)  # {'vectorizer': ..., 'clf': ...}

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return grounding_pairs, grounding_embeddings, baseline, embed_model


def run_eval(eval_path: str, data_dir: str, results_dir: str):
    eval_df = pd.read_csv(eval_path)
    grounding_pairs, grounding_embeddings, baseline, embed_model = load_artifacts(data_dir)

    print(f"Running pipeline over {len(eval_df)} eval examples...")
    results = []
    for i, row in eval_df.iterrows():
        try:
            r = run_pipeline(row["text"], embed_model, grounding_pairs, grounding_embeddings)
        except Exception as e:
            print(f"Failed on row {i}: {e}")
            r = {"message": row["text"], "intent": "ERROR", "reply": "", "auto_handle": None, "reason": str(e)}
        r["tweet_id"] = row["tweet_id"]
        r["true_intent"] = row["true_intent"]
        r["true_auto_handle"] = row["should_auto_handle"]
        results.append(r)
        if i % 20 == 0:
            print(f"  {i}/{len(eval_df)}")
        time.sleep(0.3)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{results_dir}/pipeline_results.csv", index=False)

    print("\n=== LLM classifier report ===")
    print(classification_report(results_df["true_intent"], results_df["intent"], zero_division=0))

    print("\n=== Simple baseline (TF-IDF + LogReg) report ===")
    baseline_preds = [
        classify_intent_baseline(t, baseline["vectorizer"], baseline["clf"]) for t in eval_df["text"]
    ]
    print(classification_report(eval_df["true_intent"], baseline_preds, zero_division=0))

    print("\n=== Escalation metrics ===")
    from sklearn.metrics import precision_score, recall_score
    true_escalate = ~results_df["true_auto_handle"]
    pred_escalate = ~results_df["auto_handle"]
    print(f"Recall:    {recall_score(true_escalate, pred_escalate):.3f}")
    print(f"Precision: {precision_score(true_escalate, pred_escalate):.3f}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="data/golden_eval_set.csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    run_eval(args.eval_set, args.data_dir, args.results_dir)
