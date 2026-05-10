import sys
import os

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage.db import init_db, engine
from src.storage.models import Image, Feature, SearchLog
from src.storage.vector_store import VectorStore

def verify():
    print("--- Initializing DB (Method 1) ---")
    init_db()
    print("DB initialized successfully.")
    
    from sqlalchemy import inspect
    inspector = inspect(engine)
    
    for table_name in ["images", "features", "search_logs"]:
        if table_name in inspector.get_table_names():
            print(f"Table '{table_name}': OK")
            columns = [c["name"] for c in inspector.get_columns(table_name)]
            print(f"  Columns: {columns}")
        else:
            print(f"ERROR: Table '{table_name}' does not exist!")

    print("\n--- Checking Vector Store (Method 2) ---")
    vs = VectorStore("test.index", "test_ids.npy")
    print(f"Default dimension: {vs.dim}")
    if vs.dim == 27:
        print("Dimension: OK (27)")
    else:
        print(f"ERROR: Dimension mismatch! Expected 27, got {vs.dim}")

if __name__ == "__main__":
    verify()
