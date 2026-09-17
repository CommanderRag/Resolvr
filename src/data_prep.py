"""
Data layer: filter the Twitter Customer Support dataset to one brand and
reconstruct conversation threads.

This is the slow, one-time step. Its outputs (thread_starters.csv,
resolved_threads.pkl) are checked into data/ so evaluate.py does not need
to re-run this. See README.md for the full reproduce instructions.
"""
import pandas as pd
import pickle
from tqdm import tqdm


def load_brand_data(df: pd.DataFrame, brand: str = "AmazonHelp") -> pd.DataFrame:
    """Filter the full dataset down to customer messages that this brand replied to."""
    brand_ids = df[(df["inbound"] == False) & (df["author_id"] == brand)]
    customer_tweet_ids = brand_ids["in_response_to_tweet_id"].dropna().unique()
    return df[df["tweet_id"].isin(customer_tweet_ids)]


def get_thread_starters(customer_msgs: pd.DataFrame) -> pd.DataFrame:
    """Keep only thread-starting messages (no prior context) -- the classification unit."""
    return customer_msgs[customer_msgs["in_response_to_tweet_id"].isna()]


def build_thread(start_tweet_id, df_indexed: pd.DataFrame, max_turns: int = 10) -> list:
    """Walk the reply chain forward from a starting tweet, using a tweet_id-indexed df for O(1) lookups."""
    thread = []
    current_id = start_tweet_id
    for _ in range(max_turns):
        if current_id not in df_indexed.index:
            break
        row = df_indexed.loc[current_id]
        if isinstance(row, pd.DataFrame):  # guard against duplicate tweet_ids
            row = row.iloc[0]
        thread.append({
            "tweet_id": row["tweet_id"],
            "author": row["author_id"],
            "inbound": row["inbound"],
            "text": row["text"],
        })
        raw_next = row["response_tweet_id"]
        if pd.isna(raw_next):
            break
        # response_tweet_id can be comma-separated if a tweet had multiple replies;
        # we take the first branch (see decision log in report.md)
        current_id = int(str(raw_next).split(",")[0])
    return thread


def build_all_threads(thread_starters: pd.DataFrame, df: pd.DataFrame) -> list:
    """Build resolved threads for every thread-starter. Slow: ~77k threads takes a few minutes."""
    df_indexed = df.set_index("tweet_id", drop=False)
    return [
        build_thread(tid, df_indexed)
        for tid in tqdm(thread_starters["tweet_id"], desc="Building threads")
    ]


if __name__ == "__main__":
    # Example end-to-end run. Requires twcs.csv (the raw Kaggle dataset) in data/.
    df = pd.read_csv("data/twcs.csv")
    customer_msgs = load_brand_data(df, brand="AmazonHelp")
    thread_starters = get_thread_starters(customer_msgs)
    thread_starters.to_csv("data/thread_starters.csv", index=False)

    resolved_threads = build_all_threads(thread_starters, df)
    with open("data/resolved_threads.pkl", "wb") as f:
        pickle.dump(resolved_threads, f)

    print(f"Saved {len(thread_starters)} thread starters and {len(resolved_threads)} resolved threads.")
