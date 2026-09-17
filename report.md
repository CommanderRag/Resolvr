# AI Customer Support Agent for AmazonHelp
### Hiver SDE Intern Take-Home Report

---

## 1. Problem framing

**Brand chosen:** AmazonHelp (via `@AmazonHelp` and the regional handles that show up in the data, like `@115821`, `@115830`, `@115850`). I picked it because it had high message volume, a wide but still learnable spread of recurring issue types, and long resolved threads I could actually use as grounding material for reply drafting.

**What "good" means for this system:**
- **Classification** is good if it reliably puts messages into intents that map to genuinely different resolution paths. Two messages that need different handling shouldn't usually end up in the same bucket.
- **Reply drafting** is good if it stays factually consistent with how the brand has actually resolved similar issues in the past (no invented promises, dates, or refund amounts) and reads as empathetic and on brand.
- **Escalation** is good if it almost never misses a case that truly needs a human. False negatives are expensive here. A customer with a billing dispute or a hacked account who gets auto handled badly is a real problem, so I was fine trading some over caution for that safety margin.

**What I chose not to build:**
- No live Twitter API integration. This is an offline pipeline evaluated against sampled historical data, not a deployed bot.
- No multilingual reply generation as a first class feature. Non English messages were filtered out before intent taxonomy discovery (see the decision log), though the LLM does still reply in language on its own when it happens naturally.
- No fine tuned classifier. I used prompted LLM classification plus a TF-IDF and logistic regression baseline for comparison, rather than training a custom transformer, given the time I had.
- No per message escalation reasoning beyond an intent level default policy (see the failure analysis and decision log). That's a real simplification I'm calling out on purpose, not something I missed.

---

## 2. System architecture

1. **Data layer.** Twitter Customer Support dataset filtered down to AmazonHelp. Thread starting customer messages (no prior context) were reconstructed by walking `response_tweet_id` and `in_response_to_tweet_id` chains against a `tweet_id` indexed lookup of the full dataset.
2. **Intent taxonomy discovery.** Sentence embeddings (`all-MiniLM-L6-v2`) plus K-Means clustering (k=12) over an 800 message, English only sample, followed by manually reading, merging, and relabeling the messier clusters into a final **10 intent taxonomy** (9 real intents plus an "Other" catch all).
3. **Classification.** A prompted LLM (Groq hosted `openai/gpt-oss-120b`) is given the intent definitions and classifies each message, few shot style. I compared this against a TF-IDF plus logistic regression baseline trained on the 800 message discovery set.
4. **Grounding and retrieval.** Full resolved threads (customer message plus the brand's actual reply) were embedded and indexed. For each incoming message, the top 3 nearest neighbors by cosine similarity are pulled in as grounding context.
5. **Reply drafting.** An LLM prompt combines the customer's message, its classified intent, and the 3 retrieved historical resolutions, and is instructed not to promise anything the retrieved examples don't support.
6. **Escalation decision.** An intent level `should_auto_handle` policy table, with a stated reason per intent (see the decision log for the discussion on per row overrides).

---

## 3. Results vs. baselines

**Golden evaluation set:** 188 hand labeled examples. I stratified the sample using a rough classifier over the full thread starter pool, then independently hand labeled every example for `true_intent`, `should_auto_handle`, and `escalation_reason`.

| System | Intent accuracy | Notes |
|---|---|---|
| **Trivial baseline** (always predict the most common intent, "Delivery delay") | **27.1%** | n = 188 |
| **Simple baseline** (TF-IDF + logistic regression, trained on the discovery set plus a small number of eval set derived seed examples for 3 rare classes, see decision log) | **about 34%** (weighted avg F1 around 0.30, see per class breakdown below) | n = 188 |
| **This system** (prompted LLM classifier) | **51.6%** | See confusion matrix below |

**Simple baseline, per class detail:**

| Intent | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Account security / access issue | 1.00 | 0.50 | 0.67 | 2 |
| App / device / streaming technical issue | 0.07 | 0.12 | 0.09 | 16 |
| Billing / unauthorized charge | 0.00 | 0.00 | 0.00 | 18 |
| Delivery delay | 0.43 | 0.25 | 0.32 | 51 |
| Lost / misdelivered package | 0.00 | 0.00 | 0.00 | 8 |
| Other | 0.17 | 0.47 | 0.25 | 15 |
| Positive feedback / no action needed | 1.00 | 0.57 | 0.73 | 14 |
| Refund / return request | 0.17 | 0.12 | 0.14 | 16 |
| Unable to reach support / CS unresponsive | 0.00 | 0.00 | 0.00 | 15 |
| Wrong / damaged / defective item | 0.95 | 0.55 | 0.70 | 33 |

The baseline does surprisingly well on "Positive feedback" and "Wrong / damaged / defective item," both of which were seeded into its training data directly from held out eval examples (see decision log). It scores 0.00 across the board on "Billing," "Lost / misdelivered package," and "Unable to reach support," where it had effectively no real training signal. That split tells its own story: a simple bag of words classifier can do reasonably well as long as it has some real training examples for a class, but it collapses to random guessing without them. The LLM classifier doesn't have that problem in the same way, since it has at least some understanding of all 10 categories just from reading their definitions in the prompt.

**Escalation decision performance:**

| Metric | Value |
|---|---|
| Escalation recall (catching true escalations) | **96.5%** |
| Escalation precision | **85.2%** |
| Base rate (fraction of eval set that should escalate) | 76.1% |

I deliberately biased this toward high recall. It misses only 5 of 143 true escalate cases, at the cost of over escalating on roughly 15% of cases that could have been auto handled. Since a missed escalation (say, an auto handled billing dispute or account security issue) is a lot more costly than an unnecessary human review, that tradeoff was intentional.

**Reply quality (LLM as judge, n = 188):**

| Criterion | Mean score (1 to 5) |
|---|---|
| Grounded | 2.92 |
| Correctness | 3.60 |
| Tone | 4.42 |
| Conciseness | 3.96 |

**Judge to human agreement** (30 sample manual check, Pearson correlation):

| Criterion | Correlation | n |
|---|---|---|
| Grounded | 0.01 | 30 |
| Correctness | -0.03 | 29 |
| Tone | 0.15 | 29 |
| Conciseness | 0.52 | 29 |

These correlations are weak on three of the four criteria. Only "conciseness" shows a moderate, meaningful relationship between the judge's scores and my own hand scoring. This matters: **the LLM as judge should not be treated as a reliable stand in for human judgment on "grounded," "correctness," or "tone" in this setup.** Part of the reason is probably limited score variance on both sides. My own hand scores clustered heavily in the 4 to 5 range on this sample, and the judge's "tone" scores were similarly bunched near the top of the scale (mean of 4.42 above), which mechanically suppresses correlation even when both raters broadly agree the replies are decent. A bigger, more score diverse human sample, and probably a more discriminating rubric with anchored descriptions per score, would be needed before I'd trust the automated "grounded," "correctness," and "tone" numbers as a headline result. Separately, `judge_reply`'s text parsing failed to pull out a score for "conciseness" on 18 of 188 rows and "correctness" on 2 of 188 rows, most likely from inconsistent formatting in the LLM's output. The correlations above only use the rows where parsing succeeded.

---

## 4. Failure analysis, top 5 modes

**1. "Lost / misdelivered package" over triggers on "Delivery delay" cases (precision 0.24, recall 0.62)**
The model predicts this intent 21 times but only 5 of those are correct, and 7 of the wrong ones are actually "Delivery delay." My guess is the two definitions are just semantically close ("marked delivered but I don't have it" vs. "hasn't arrived yet") and the model leans toward the more dramatic sounding label whenever it's unclear whether tracking shows "delivered" or just "in transit."
*Example:* `"@115821 paid for next day delivery.. had it confirmed and alas, no package #nothappy"`. There's no explicit "delivered" status mentioned here, and the model got this one right as Delivery delay, but several very similar phrasings elsewhere got misrouted.

**2. "Other" gets used as a low confidence fallback (precision 0.21)**
The model predicts "Other" 33 times, but only 7 of those are genuinely unclassifiable. It's absorbing real "Wrong / damaged item" cases (8 of them) and "Billing" cases (3 of them). My guess is that when a message is terse, sarcastic, or doesn't use obvious issue describing keywords, the model just defaults to "Other" instead of committing to the closest real intent.

**3. "Unable to reach support / CS unresponsive" has very low recall (0.20)**
True cases of this intent end up scattered across "Other," "Delivery delay," and "Lost / misdelivered." These messages usually describe an underlying issue and a complaint about support responsiveness at the same time (something like "delivery boy not picking my phone, please resolve"), and the model seems to latch onto the underlying issue and miss the responsiveness complaint, which is arguably the more urgent signal for deciding whether to escalate.

**4. Rare intents like "Account security / access issue" have no reliable signal (0 of 2 correct)**
With only 2 true examples in the eval set this isn't statistically meaningful on its own, but it does point to a real coverage gap. The discovery clustering only surfaced around 5 security flavored messages out of 800 sampled, so the model has seen very few real examples of this pattern to calibrate against.

**5. Escalation reasoning is intent level, not message level**
Every message with a given predicted intent gets the exact same `should_auto_handle` value and the exact same stated reason. So the "Decide" step right now doesn't add any judgment beyond a lookup table. A real system would want message level signals, things like repeat contact history, how intense the sentiment is, or thread length, to tell apart a calm first time delivery question from an angry fourth follow up on the same order.

---

## 5. What's misleading about my headline number

The 51.6% accuracy figure needs a few caveats to be read fairly.

- **The eval set's own labels were bootstrapped from a rough classifier**, not sampled purely at random. Every label was independently hand verified, but which messages got sampled in the first place was guided by a baseline model's predictions, which could quietly under represent the message types that baseline model itself struggled with.
- **Ten unevenly sized classes make "accuracy" a blunt instrument.** Macro F1 (0.45) is meaningfully lower than the weighted average (0.53), which means the model does worse on rare intents than the headline number suggests, since common intents like Delivery delay and Wrong item dominate the overall accuracy calculation.
- **"Other" and "Lost / misdelivered" together account for a large share of the total errors.** If a production system just routed those two categories to a human reviewer by default instead of trying to auto classify them, effective accuracy on the remaining 7 intents would look considerably better than 51.6% suggests.
- **Escalation recall (96.5%) is honestly the number that matters most for safety**, and it's a lot stronger than the intent accuracy number. A system that gets the intent wrong but still correctly escalates the message avoids most of the real world harm, so the intent accuracy headline actually undersells how safe the system is in practice.
- **The taxonomy itself was hand merged from ambiguous clusters** (see decision log), so some of the "misclassifications" above are genuinely defensible alternate readings of an ambiguous message, not clear cut model errors.
- **The simple baseline comparison, 34% vs 51.6%, isn't a fully clean comparison.** Two of the baseline's best performing classes were partly seeded with held out golden eval set examples (see decision log), because there was no clustering derived training data available for them. The real gap between "a simple trained classifier with adequate training data" and the LLM approach is probably narrower than 34% vs 51.6% suggests, though it's hard to say in which direction, since we don't have a clean apples to apples baseline number for classes the baseline never had organic training signal for.
- **The LLM as judge reply quality scores (like "grounded: 2.92 out of 5") should be read with real skepticism.** Judge to human agreement was weak, near zero or even slightly negative, for grounded, correctness, and tone. Only conciseness showed a moderate, trustworthy correlation. That means the headline reply quality numbers for those three dimensions aren't well validated against human judgment and shouldn't be treated as precise measurements.

---

## 6. What I'd do with one more week

1. Tighten the "Delivery delay" vs. "Lost / misdelivered package" definitions with an explicit rule (for example, whether a "delivered" status is present or not) and re-run just those two categories to see how much it helps.
2. Add message level features to the escalation decision, things like a sentiment score, repeat contact detection from thread history, or explicit urgency keywords, instead of a flat intent level policy.
3. Fine tune a small classifier, or add more few shot examples per intent to the LLM prompt, specifically targeting the three weakest intents identified above.
4. Expand the golden eval set past 188 examples, oversampling the rare intents (Account security, Unable to reach support) so their precision and recall numbers are actually statistically meaningful.
5. Add a confidence or abstention mechanism, so that when the classifier's top intent is a close call it routes to human review instead of being forced into a single label.

---

## 7. Decision log

- Chose AmazonHelp over other brands for its volume and how repetitive the issue types were.
- Used thread starting messages only (no prior context) as the classification unit, separate from the full resolved threads used for RAG grounding.
- Filtered non English messages out of the taxonomy discovery clustering step, since the sentence transformer model was conflating language with topic. Treated multilingual support as out of scope for this version.
- Used K-Means (k=12) for intent discovery instead of HDBSCAN, mainly for simplicity and because I didn't mind pre-specifying k.
- Hand merged and relabeled two persistently mixed clusters at the message level rather than tuning k further, since clustering couldn't cleanly separate intents that are semantically close but functionally different, like "wrong item" and "refund request."
- Landed on a 9 intent plus "Other" taxonomy after two rounds of cluster inspection.
- Used the root cause of a message, not the requested resolution, to break ties when a message touched more than one intent during manual labeling.
- Built the golden eval set through stratified sampling, using a baseline classifier's predictions purely as a sampling tool and not as ground truth. Every sampled example was independently hand labeled.
- Set a minimum of 15 examples per intent when sampling the eval set to guarantee coverage of rare intents, then topped up to the target total with additional random samples.
- Adopted a conservative escalation policy, defaulting to escalate unless an intent is clearly low risk, since missed escalations are costlier than unnecessary ones.
- Escalation reasons are intent level defaults for this version, explicitly not per message judgments.
- Chose a free, Groq hosted open model (`openai/gpt-oss-120b`) over a paid API to avoid cost during development. Had to raise `max_tokens` substantially above what a non reasoning model would need, since this model spends tokens on internal reasoning before producing its answer.
- Capped the RAG grounding pool at 3,000 resolved threads out of a much larger pool, for practical embedding time, rather than indexing the full dataset.
- Took the first branch whenever a tweet had multiple replies (a comma separated `response_tweet_id`), rather than modeling out alternate conversation branches.
- The baseline classifier's training set was missing 3 of 10 intent categories entirely, because of non reproducible sampling across notebook re-runs. Different random seeds and sample sizes produced different cluster contents each time, so a taxonomy hand labeled against one run's clusters didn't carry over to another run. Rather than keep chasing exact reproducibility, I patched the training set with a small number of held out golden eval set examples for the 3 missing classes. I'm calling this out explicitly since it makes the baseline's performance on those specific 3 classes somewhat optimistic, because it was partly evaluated on examples similar to its own training data.
- `judge_reply`'s output parsing isn't fully robust to variation in the LLM's response formatting, which caused roughly 1 to 10 percent missing scores per criterion, worst for "conciseness." Judge to human agreement was computed only on the rows with complete parsed output rather than throwing out the whole judge evaluation.
- Treated the reply drafting LLM's occasional habit of replying in the customer's own language (like replying in Spanish to a Spanish message) as an acceptable side effect rather than a bug, even though it wasn't something I explicitly built for.

---

*A note on completeness: the numbers in this report come from a single run of the pipeline over the 188 row golden eval set, combined with one round of manual human scoring on a 30 row sample for judge agreement. Given more time I'd want to repeat both with a couple of different random seeds to see how stable these numbers actually are.*
