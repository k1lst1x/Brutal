from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel


class Stats(BaseModel):
    total_customers_in_history: int
    model_version: str
    threshold: float
    num_features: int


class Models(BaseModel):  # индивидуальные скоринги моделей
    catboost: float
    xgboost: float
    lightgbm: float
    anomaly: float


class TopFeature(BaseModel):
    feature: str
    shap_value: float


class TransactionInput(BaseModel):
    """
    Сырые поля, которые приходят от фронта / других сервисов.
    """
    cst_dim_id: int
    amount: float
    direction: str
    transdatetime: datetime

    # доп. поля по желанию
    id: Optional[int] = None
    behavioral_patterns: Dict[str, Any] = None
    target: Optional[int] = None


class TransactionOutput(BaseModel):
    """
    То, что возвращает /predict.
    Эти поля нужны и Streamlit, и React.
    """
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    alerts: List[str]
    processing_time_ms: float
    model_version: str
    threshold_used: float
    individual_scores: Models

    # --- новое поле для React (график SHAP) ---
    top_features: Optional[List[TopFeature]] = None
