"""
Intent taxonomy and escalation policy for the Resolvr pipeline.

These definitions were derived by clustering ~800 English-language customer
messages (sentence embeddings + K-Means, k=12), then manually reading and
merging ambiguous clusters. See notebook/exploration.ipynb for the full
discovery process.
"""

INTENT_DEFINITIONS = {
    "Delivery delay": "Item has not arrived by the promised/expected date, but is still presumed in transit.",
    "Lost / misdelivered package": "Tracking shows delivered (or long overdue) but the customer never received it, or it went to the wrong address.",
    "Wrong / damaged / defective item": "Customer received an item that is broken, defective, or not what they ordered.",
    "Refund / return request": "Customer wants to return an item or is following up on a refund already in process.",
    "Billing / unauthorized charge": "Customer was charged incorrectly, twice, or for something they didn't order.",
    "Account security / access issue": "Account was hacked, locked, or accessed without consent.",
    "App / device / streaming technical issue": "A bug, crash, or feature failure in the app/device/streaming product.",
    "Unable to reach support / CS unresponsive": "Customer is complaining about not getting a response from support itself.",
    "Positive feedback / no action needed": "Praise, thanks, or a comment that doesn't require a support action.",
    "Other": "Doesn't clearly fit any category above.",
}

# Whether an intent can be safely auto-handled without human review.
# Biased toward escalation: missing a true escalation is costlier than an
# unnecessary human review. See report.md, section 3, for the recall/precision
# tradeoff this produces.
AUTO_HANDLE_POLICY = {
    "Delivery delay": False,
    "Lost / misdelivered package": False,
    "Wrong / damaged / defective item": False,
    "Refund / return request": False,
    "Billing / unauthorized charge": False,
    "Account security / access issue": False,
    "App / device / streaming technical issue": True,
    "Unable to reach support / CS unresponsive": False,
    "Positive feedback / no action needed": True,
    "Other": False,
}

DEFAULT_MODEL = "openai/gpt-oss-120b"
