from pydantic import BaseModel
from typing import List

class SingleFeatures(BaseModel):
    duration: float
    src_bytes: float
    dst_bytes: float
    count: float
    srv_count: float
    wrong_fragment: float

class BulkFeatures(BaseModel):
    items: List[SingleFeatures]
