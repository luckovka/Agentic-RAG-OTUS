from pathlib import Path
from qdrant_client import QdrantClient

for db_path in [Path("qdrant_data"), Path("src/qdrant_data")]:
    client = QdrantClient(path=str(db_path))

    print(f"\nБаза: {db_path.resolve()}")

    for collection in client.get_collections().collections:
        info = client.get_collection(collection.name)

        print("Коллекция:", collection.name)
        print("Точек:", info.points_count)
        print("Векторы:", info.config.params.vectors)

    client.close()