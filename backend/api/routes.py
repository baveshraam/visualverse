"""
API Routes for VisualVerse
Separate route definitions for cleaner architecture
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

router = APIRouter(prefix="/api", tags=["api"])


class TextInput(BaseModel):
    text: str
    mode: Optional[str] = "auto"


class ProcessingResult(BaseModel):
    mode: str
    title: str
    summary: str
    comic_data: Optional[List[Dict[str, Any]]] = None
    mindmap_data: Optional[Dict[str, Any]] = None


@router.get("/health")
async def health():
    return {"status": "healthy"}
