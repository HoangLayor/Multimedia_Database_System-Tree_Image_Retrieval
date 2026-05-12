"""
db.py — Quản lý kết nối CSDL (SQLAlchemy)
"""
import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Mặc định dùng SQLite để dev nhanh, có thể thay đổi qua biến môi trường cho PostgreSQL
# DATABASE_URL = "postgresql://user:password@localhost/tree_db"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tree_search.db")

engine = create_engine(
    DATABASE_URL, 
    # SQLite yêu cầu check_same_thread=False để dùng trong context đa luồng
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_session():
    """Cung cấp database session cho một khối lệnh."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def init_db():
    """Khởi tạo các bảng (dùng khi setup ban đầu)."""
    from .models import Base
    Base.metadata.create_all(bind=engine)
