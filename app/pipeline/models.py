"""
AI Document Classification System
Developed by Rohan Shekokar
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from PIL import Image

@dataclass
class TextBlock:
    text: str
    bbox: List[int]
    confidence: float

@dataclass
class LayoutRegion:
    region_type: str
    bbox: List[int]
    text_blocks: List[TextBlock] = field(default_factory=list)
    html_content: Optional[str] = None

@dataclass
class PageData:
    page_number: int
    image: Image.Image
    raw_text_blocks: List[TextBlock] = field(default_factory=list)
    layout_regions: List[LayoutRegion] = field(default_factory=list)

@dataclass
class DocumentContext:
    file_name: str
    pages: List[PageData] = field(default_factory=list)
    aggregated_text: str = ""
    
@dataclass
class ClassificationResult:
    document_type: str
    writing_type: str
    language: str
    summary: str
    confidence: float
