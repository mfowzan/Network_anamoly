from pydantic import BaseModel
from typing import List

class SingleFeatures(BaseModel):
    duration: float = 0
    src_bytes: float = 0
    dst_bytes: float = 0
    count: float = 0
    srv_count: float = 0
    wrong_fragment: float = 0
    serror_rate: float = 0
    srv_serror_rate: float = 0
    rerror_rate: float = 0
    srv_rerror_rate: float = 0
    same_srv_rate: float = 0
    diff_srv_rate: float = 0
    dst_host_count: float = 0
    dst_host_srv_count: float = 0
    dst_host_same_srv_rate: float = 0
    dst_host_diff_srv_rate: float = 0

class BulkFeatures(BaseModel):
    items: List[SingleFeatures]
