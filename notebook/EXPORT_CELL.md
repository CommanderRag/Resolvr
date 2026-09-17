# One-time export cell

Add this as a new cell at the end of your Colab notebook (CS_Agent.ipynb), run it once,
then download the 4 files it produces and place them in resolvr/data/.

```python
import pickle

with open('grounding_pairs.pkl', 'wb') as f:
    pickle.dump(grounding_pairs, f)

with open('grounding_embeddings.pkl', 'wb') as f:
    pickle.dump(grounding_embeddings, f)

with open('baseline_classifier.pkl', 'wb') as f:
    pickle.dump({'vectorizer': vectorizer, 'clf': clf}, f)

# also grab your golden eval set and results, if not already downloaded
import shutil
shutil.copy('golden_eval_clean_updated_taxonomy.csv', 'golden_eval_set.csv')

print("Done. Download these 4 files from the Colab file browser:")
print(" - grounding_pairs.pkl")
print(" - grounding_embeddings.pkl")
print(" - baseline_classifier.pkl")
print(" - golden_eval_set.csv")
```

After downloading, your data/ folder should contain:

```
data/
├── golden_eval_set.csv
├── grounding_pairs.pkl
├── grounding_embeddings.pkl
└── baseline_classifier.pkl
```

results/ should contain your already-generated outputs:
```
results/
├── pipeline_results.csv
└── judge_scores.csv
```
