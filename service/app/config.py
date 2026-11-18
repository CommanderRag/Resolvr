import os

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://resolvr:resolvr@localhost:5432/resolvr")
# Where the existing Resolvr code and artifacts live (repo-root relative by default).
RESOLVR_SRC = os.environ.get("RESOLVR_SRC", "../src")
DATA_DIR = os.environ.get("DATA_DIR", "../data")
