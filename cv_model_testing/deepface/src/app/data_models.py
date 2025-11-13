# src/data_models.py
from typing import List, Tuple, Optional
from pydantic import BaseModel
import datetime

class Detection(BaseModel):
    bbox: Tuple[int,int,int,int]
    confidence: float

class TrackedDetection(Detection):
    track_id: int

class AnnotatedDetection(TrackedDetection):
    age: Optional[float]
    gender: Optional[str]
    timestamp: datetime.datetime

class LogEntry(BaseModel):
    track_id: int
    age: Optional[float]
    gender: Optional[str]
    timestamp: datetime.datetime
    frame_index: int
    bbox: Tuple[int,int,int,int]
    confidence: float
