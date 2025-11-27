from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class Stats(BaseModel):
    total_customers_in_history: int
    model_version: str
    threshold: float
    num_features: int

class Models(BaseModel):  # model scores
    catboost: float
    xgboost: float
    lightgbm: float
    anomaly: float


class TransactionInput(BaseModel):
    cst_dim_id: int
    transdatetime: str
    amount: float
    direction: str

    id: Optional[int] = None
    behavioral_patterns: Dict[str, Any] = None


class TransactionOutput(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    alerts: List[str]
    processing_time_ms: float
    model_version: str
    threshold_used: float
    individual_scores: Models
