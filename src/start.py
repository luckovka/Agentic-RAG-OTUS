import os
import sys
import subprocess

from .database import get_client
from .config import e5_collection


def check_database() -> bool:
    client = get_client()

    try:
        if not client.collection_exists(e5_collection):
            return False

        count = client.count(
            collection_name=e5_collection,
            exact=True
        ).count

        print(f"Points in Qdrant: {count}", flush=True)

        return count > 0

    finally:
        client.close()
        get_client.cache_clear()


def main():
    if not check_database():
        print("Creating knowledge base...", flush=True)

        subprocess.run(
            [sys.executable, "-m", "src.ingest"],
            check=True
        )
    else:
        print("Knowledge base is ready!", flush=True)

    print("Starting Cthulhu...", flush=True)

    os.execvp(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/ui.py",
            "--server.address=0.0.0.0"
        ]
    )


if __name__ == "__main__":
    main()