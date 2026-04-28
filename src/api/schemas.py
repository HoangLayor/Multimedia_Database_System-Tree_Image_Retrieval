"""
schemas.py — Pydantic models cho request/response API
"""
from pydantic import BaseModel


class SearchResult(BaseModel):
    rank: int
    image_id: int
    filename: str
    species: str | None
    common_name: str | None
    age_class: str | None
    similarity: float   # [0, 1], cao hơn = giống hơn
    filepath: str


class SearchResponse(BaseModel):
    query_filename: str
    results: list[SearchResult]
    execution_ms: float
    total_images_searched: int
