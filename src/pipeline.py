"""
Core support-agent pipeline: classify -> retrieve grounding -> draft reply -> decide escalation.

Requires a GROQ_API_KEY environment variable. See README.md for setup.
"""
import os
import re
import numpy as np
from groq import Groq

from constants import INTENT_DEFINITIONS, AUTO_HANDLE_POLICY, DEFAULT_MODEL

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def clean_text(t: str) -> str:
    """Strip @mentions and URLs, collapse whitespace."""
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'http\S+', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def classify_intent(message_text: str, model: str = DEFAULT_MODEL) -> str:
    """Classify a customer message into one of the taxonomy intents via a prompted LLM call."""
    intent_list = "\n".join(f"- {k}: {v}" for k, v in INTENT_DEFINITIONS.items())
    prompt = f"""Classify this customer support message into exactly one intent.

Intents:
{intent_list}

Message: "{message_text}"

Respond with ONLY the intent name, exactly as written above."""

    resp = client.chat.completions.create(
        model=model,
        max_tokens=300,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )
    predicted = resp.choices[0].message.content.strip()
    return predicted if predicted in INTENT_DEFINITIONS else "Other"


def classify_intent_baseline(message_text: str, vectorizer, clf) -> str:
    """TF-IDF + Logistic Regression baseline classifier, for comparison against the LLM classifier."""
    X = vectorizer.transform([clean_text(message_text)])
    return clf.predict(X)[0]


def retrieve_grounding(message_text: str, embed_model, grounding_pairs, grounding_embeddings, top_k: int = 3):
    """Retrieve the top_k most similar historical (issue, resolution) pairs by cosine similarity."""
    query_emb = embed_model.encode([clean_text(message_text)])[0]

    if grounding_embeddings.ndim == 1:
        return grounding_pairs[:1]

    norms = np.linalg.norm(grounding_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
    sims = grounding_embeddings @ query_emb / norms

    k = min(top_k, len(sims))
    top_idx = np.argsort(sims)[-k:][::-1]
    return [grounding_pairs[i] for i in top_idx]


def draft_reply(message_text: str, intent: str, grounding: list, model: str = DEFAULT_MODEL) -> str:
    """Draft a reply grounded in retrieved historical resolutions."""
    grounding_text = "\n\n".join(
        f"Similar past issue: {g['issue']}\nHow it was resolved: {g['resolution']}"
        for g in grounding
    )
    prompt = f"""You are a customer support agent for this brand. Draft a reply to the customer below.

Customer's intent: {intent}
Customer's message: "{message_text}"

Historical examples of how similar issues were resolved:
{grounding_text}

Write a short, empathetic, on-brand reply consistent with how similar issues were actually resolved. Do not promise anything not supported by the historical examples."""

    resp = client.chat.completions.create(
        model=model,
        max_tokens=500,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def judge_reply(message: str, reply: str, model: str = DEFAULT_MODEL) -> dict:
    """LLM-as-judge: score a reply 1-5 on groundedness, correctness, tone, and conciseness."""
    prompt = f"""Rate this customer support reply on a 1-5 scale for each criterion.

Customer message: "{message}"
Agent reply: "{reply}"

Criteria:
- grounded: does it sound consistent with real past resolutions, not generic?
- correctness: does it address the actual issue?
- tone: is it empathetic and professional?
- conciseness: is it appropriately brief?

Respond in this exact format:
grounded: <1-5>
correctness: <1-5>
tone: <1-5>
conciseness: <1-5>"""

    resp = client.chat.completions.create(
        model=model,
        max_tokens=100,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content

    scores = {}
    for line in text.strip().split('\n'):
        if ':' in line:
            k, v = line.split(':', 1)
            try:
                scores[k.strip().lower()] = int(v.strip())
            except ValueError:
                pass
    return scores


def decide_escalation(intent: str, message_text: str) -> tuple:
    """Look up the auto-handle policy and reason for a classified intent."""
    auto_handle = AUTO_HANDLE_POLICY.get(intent, False)
    reason = INTENT_DEFINITIONS.get(intent, "Unclassified intent")
    return auto_handle, reason


def run_pipeline(message_text: str, embed_model, grounding_pairs, grounding_embeddings) -> dict:
    """Full pipeline: classify -> retrieve -> draft -> decide, for a single message."""
    intent = classify_intent(message_text)
    grounding = retrieve_grounding(message_text, embed_model, grounding_pairs, grounding_embeddings)
    reply = draft_reply(message_text, intent, grounding)
    auto_handle, reason = decide_escalation(intent, message_text)

    return {
        "message": message_text,
        "intent": intent,
        "reply": reply,
        "auto_handle": auto_handle,
        "reason": reason,
        "grounding_used": [g["issue"] for g in grounding],
    }
