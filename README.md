# Resolvr

An AI customer support agent built and evaluated on real, messy customer support
data. Given an incoming customer message, it classifies the intent, retrieves
similar past issues the brand actually resolved, drafts a grounded reply, and
decides whether the message can be auto-handled or needs to go to a human.

Built for AmazonHelp using the [Twitter Customer Support dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).

Full writeup, including baselines, failure analysis, and known limitations: [report.md](report.md)

## What it does

1. **Classify** an incoming message into one of 10 support intents, discovered by clustering real customer messages rather than invented up front.
2. **Retrieve** the most similar past issues this brand has actually resolved, using sentence embeddings over resolved conversation threads.
3. **Draft** a reply grounded in those retrieved resolutions, not invented from scratch.
4. **Decide** whether the message can be auto-handled or should be escalated to a human, with a stated reason.

## Results

| System | Intent accuracy |
|---|---|
| Trivial baseline (always predict most common intent) | 27.1% |
| Simple baseline (TF-IDF + Logistic Regression) | ~34% |
| This pipeline (prompted LLM) | 51.6% |

Escalation decision: 96.5% recall, 85.2% precision on catching messages that truly need a human.

See [report.md](report.md) for the full breakdown, confusion matrix, failure analysis, and an honest discussion of what these numbers do and don't mean.

## Reproduce the results (< 15 minutes)

This runs the pipeline over the pre-built 188-row golden evaluation set using
precomputed grounding embeddings and a pre-trained baseline classifier. It does
not rebuild the thread data or the taxonomy from scratch.

```bash
git clone https://github.com/yourusername/resolvr.git
cd resolvr
pip install -r requirements.txt
cp .env.example .env   # add your Groq API key (free tier: console.groq.com)
export $(cat .env | xargs)
python src/evaluate.py
```

This prints the classification report for both the LLM pipeline and the simple
baseline, plus escalation precision/recall, and writes `results/pipeline_results.csv`.

## Full pipeline (slow, optional)

Rebuilding the thread data and taxonomy from the raw dataset takes significantly
longer (downloading ~3M tweets, reconstructing ~77k threads, embedding and
clustering for intent discovery). This is not required to verify the results
above, since the outputs are already checked into `data/`.

```bash
python src/data_prep.py
```

See `notebook/CS_Agent.ipynb` for the full exploratory process, including the
clustering and manual taxonomy work, and `notebook/EXPORT_CELL.md` for how the
artifacts in `data/` were produced.

## Project structure

```
resolvr/
├── README.md
├── report.md                 # full writeup: baselines, failure analysis, decision log
├── requirements.txt
├── .env.example
├── data/
│   ├── golden_eval_set.csv        # 188 hand-labeled examples
│   ├── grounding_pairs.pkl        # (issue, resolution) pairs for retrieval
│   ├── grounding_embeddings.pkl
│   └── baseline_classifier.pkl    # TF-IDF vectorizer + trained LogisticRegression
├── results/
│   ├── pipeline_results.csv
│   └── judge_scores.csv
├── src/
│   ├── constants.py           # intent taxonomy + escalation policy
│   ├── pipeline.py            # classify, retrieve, draft, decide
│   ├── data_prep.py           # thread reconstruction (slow path)
│   └── evaluate.py            # runs the pipeline over the eval set (fast path)
└── notebook/
    ├── CS_Agent.ipynb         # full exploratory process
    └── EXPORT_CELL.md
```

## Known limitations

Documented in detail in [report.md](report.md), but briefly:
- Intent accuracy (51.6%) is a meaningful improvement over baselines but far from perfect, with specific, understood confusion between "Delivery delay" and "Lost / misdelivered package."
- Escalation decisions are currently intent-level defaults, not per-message judgments.
- The LLM-as-judge reply quality scores showed weak correlation with human scoring on 3 of 4 rubric dimensions and should not be over-trusted as precise measurements.
