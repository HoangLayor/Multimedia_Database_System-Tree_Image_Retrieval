import sys
from pathlib import Path
import numpy as np

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.features.extractor import extract

def test_real_images():
    raw_dir = Path("data/raw")
    test_images = ["1.jpg", "10.jpg", "100.jpg", "101.jpg", "102.jpg"]
    
    print(f"{'Image':<10} | {'Dim':<5} | {'Feature Summary (first 5 values)':<40}")
    print("-" * 65)
    
    for img_name in test_images:
        img_path = raw_dir / img_name
        if not img_path.exists():
            print(f"{img_name:<10} | {'N/A':<5} | File not found")
            continue
            
        try:
            vector = extract(img_path)
            # Lấy 5 giá trị đầu tiên để hiển thị tóm tắt
            summary = ", ".join([f"{v:.4f}" for v in vector[:5]])
            print(f"{img_name:<10} | {len(vector):<5} | {summary}...")
        except Exception as e:
            print(f"{img_name:<10} | {'Err':<5} | {e}")

if __name__ == "__main__":
    test_real_images()
