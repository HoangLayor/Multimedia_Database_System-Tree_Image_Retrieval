import os
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
from src.storage.db import init_db, get_session
from src.storage.models import Image

from src.storage.s3 import MinioStorage

def preprocess():
    # 1. Load config
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    target_size = tuple(config["data"]["image_size"]) # (512, 512)
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Init DB and MinIO
    init_db()
    storage = MinioStorage(config)
    
    # 3. Process images
    image_files = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png"))
    print(f"Bắt đầu tiền xử lý và upload {len(image_files)} ảnh lên MinIO...")
    
    with get_session() as session:
        for img_path in tqdm(image_files):
            # Đọc và resize
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_resized = cv2.resize(img, target_size)
            
            # Lưu tạm vào processed để upload
            out_path = processed_dir / img_path.name
            cv2.imwrite(str(out_path), img_resized)
            
            # Upload lên MinIO
            s3_key = storage.upload_file(out_path)
            
            # Thêm vào DB nếu chưa có
            existing = session.query(Image).filter_by(filename=img_path.name).first()
            if not existing:
                new_img = Image(
                    filename=img_path.name,
                    file_path=s3_key if s3_key else str(out_path), # Lưu S3 Key
                    width=target_size[0],
                    height=target_size[1],
                    species="Unknown",
                    common_name="Chưa xác định",
                    age_class="N/A"
                )
                session.add(new_img)
            else:
                # Cập nhật đường dẫn nếu đã có
                existing.file_path = s3_key if s3_key else existing.file_path
        
        session.commit()
    
    print("=== Tiền xử lý hoàn tất! ===")
    print(f"Ảnh đã được lưu vào: {processed_dir}")
    print("Bây giờ bạn có thể chạy: python scripts/build_index.py")

if __name__ == "__main__":
    preprocess()
