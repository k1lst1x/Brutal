from typing import List

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from config import MODEL_DIR
from dtos import TransactionInput, TransactionOutput, Stats
from model import FraudDetectionAPI

router = APIRouter()

fraud_detector = FraudDetectionAPI(MODEL_DIR)

@router.post("/predict", response_model=TransactionOutput)
async def predict_fraud(transaction: TransactionInput):
    """ Predict single transaction """
    try:
        """test_transaction = {
            'cst_dim_id': transaction.cst_dim_id,
            'transdatetime': transaction.transdatetime,
            'amount': transaction.amount,
            'direction': transaction.direction,
        }"""
        result = await run_in_threadpool(fraud_detector.predict_single_transaction, transaction)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=List[TransactionOutput])
async def predict_batch(transactions: List[TransactionInput]):
    """ Predict several transactions """
    try:
        result = await run_in_threadpool(fraud_detector.predict_batch, transactions)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=Stats)
async def get_stats():
    """ Get model statistics """
    try:
        result = await run_in_threadpool(fraud_detector.get_stats)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health_check():
    """ Health check for model """
    return {"status": "healthy", **fraud_detector.get_stats()}
