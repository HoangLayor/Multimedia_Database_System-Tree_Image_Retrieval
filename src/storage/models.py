"""
models.py — SQLAlchemy ORM models
Thiết kế dựa trên đặc tả hệ thống Multimedia Database.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    ForeignKey, Index, JSON
)
# Sử dụng ARRAY của Postgres nếu cần, hoặc JSON/Pickle cho tính di động
from sqlalchemy.orm import DeclarativeBase, relationship

class Base(DeclarativeBase):
    pass

class Image(Base):
    """
    Bảng 1: images
    Lưu thông tin siêu dữ liệu của ảnh.
    """
    __tablename__ = "images"

    id          = Column(Integer, primary_key=True)
    file_path   = Column(String(512), nullable=False)   # Đường dẫn tới file vật lý (MinIO/local)
    width       = Column(Integer)
    height      = Column(Integer)
    created_at  = Column(DateTime, default=datetime.utcnow)

    # Thông tin bổ sung (Metadata tùy chọn)
    filename    = Column(String(255), unique=True)
    species     = Column(String(100))
    common_name = Column(String(100))
    age_class   = Column(String(20))

    feature = relationship("Feature", back_populates="image", uselist=False)

    def __repr__(self):
        return f"<Image id={self.id} path={self.file_path}>"

class Feature(Base):
    """
    Bảng 2: features
    Lưu vector đặc trưng phục vụ truy xuất (Phương án 1).
    """
    __tablename__ = "features"

    image_id   = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), primary_key=True)
    
    # vector FLOAT[] - Ở đây dùng JSON để tương thích cả SQLite và Postgres (Method 1)
    # Nếu dùng Postgres thật sự có thể dùng sqlalchemy.dialects.postgresql.ARRAY(Float)
    vector     = Column(JSON, nullable=False) 
    
    cluster_id = Column(Integer, nullable=True) # Dùng cho partitioning/indexing

    image = relationship("Image", back_populates="feature")

    def to_vector(self):
        """Chuyển đổi từ DB format sang numpy array."""
        import numpy as np
        return np.array(self.vector, dtype=np.float32)

class SearchLog(Base):
    """
    Bảng 3: search_logs
    Phục vụ đánh giá và đo hiệu năng hệ thống.
    """
    __tablename__ = "search_logs"

    id          = Column(Integer, primary_key=True)
    query_time  = Column(DateTime, default=datetime.utcnow)
    latency     = Column(Float)  # ms
    top_k       = Column(Integer)
    
    # Thông tin thêm cho việc audit
    query_image_path = Column(String(512))
    result_ids       = Column(JSON) # Lưu list các ID kết quả trả về

    def __repr__(self):
        return f"<SearchLog id={self.id} latency={self.latency}ms>"
