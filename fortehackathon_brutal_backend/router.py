from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Response
from starlette.concurrency import run_in_threadpool

import pandas as pd
from io import StringIO, BytesIO

from config import MODEL_DIR
from dtos import TransactionInput, TransactionOutput, Stats
from model import FraudDetectionAPI

router = APIRouter()

# Загружаем прод-модель
fraud_detector = FraudDetectionAPI(MODEL_DIR)


@router.post("/predict", response_model=TransactionOutput)
async def predict_fraud(transaction: TransactionInput):
    """Предсказывает фрод по одной транзакции (online-режим)."""
    try:
        result = await run_in_threadpool(
            fraud_detector.predict_single_transaction,
            transaction,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=List[TransactionOutput])
async def predict_batch(transactions: List[TransactionInput]):
    """Предсказывает фрод по списку транзакций (JSON batch)."""
    try:
        result = await run_in_threadpool(
            fraud_detector.predict_batch,
            transactions,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk_predict", response_class=Response)
async def bulk_predict(file: UploadFile = File(...)):
    """
    Batch-режим для data scientist'ов:
    принимает CSV, прогоняет через модель и возвращает CSV с колонками:
      - fraud_score      (вероятность фрода)
      - prediction       (0/1)
      - risk_level       (низкий/средний/высокий)
      - model_version
      - threshold_used
      - score_catboost
      - score_xgboost
      - score_lightgbm
      - anomaly_score

    Ожидаемые колонки во входном файле:
      cst_dim_id, amount, direction, transdatetime

    Поддерживаются:
      • обычные utf-8 csv (разделитель по умолчанию)
      • исходный банковский csv в cp1251 с разделителем ';' и skiprows=1.
    """
    try:
        contents = await file.read()

        # 1) сначала пробуем как utf-8
        try:
            s = contents.decode("utf-8")
            df = pd.read_csv(StringIO(s))
        except UnicodeDecodeError:
            # 2) fallback — банковский формат cp1251 + ';' + skiprows=1
            df = pd.read_csv(
                BytesIO(contents),
                encoding="cp1251",
                sep=";",
                skiprows=1,
                low_memory=False,
            )

        required_cols = {"cst_dim_id", "amount", "direction", "transdatetime"}
        missing = required_cols - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Отсутствуют обязательные колонки: "
                    f"{', '.join(sorted(missing))}"
                ),
            )

        # ---- Приведение типов и очистка ----
        # cst_dim_id -> число
        df["cst_dim_id"] = pd.to_numeric(df["cst_dim_id"], errors="coerce")

        # amount как в твоём пайплайне (убрать кавычки, пробелы, запятые)
        df["amount"] = (
            df["amount"]
            .astype(str)
            .str.replace('"', "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        # direction и transdatetime как строки
        df["direction"] = df["direction"].astype(str)
        df["transdatetime"] = df["transdatetime"].astype(str)

        # выкидываем строки, где что-то критично пропало
        df_before = len(df)
        df = df.dropna(
            subset=["cst_dim_id", "amount", "direction", "transdatetime"]
        ).reset_index(drop=True)
        df_after = len(df)

        if df_after == 0:
            raise HTTPException(
                status_code=400,
                detail="После очистки данных не осталось ни одной валидной строки (все с NaN / пустыми полями).",
            )

        # ---- Собираем TransactionInput ----
        transactions: List[TransactionInput] = []
        for row in df.itertuples(index=False):
            trans = TransactionInput(
                cst_dim_id=int(getattr(row, "cst_dim_id")),
                amount=float(getattr(row, "amount")),
                direction=str(getattr(row, "direction")),
                # важно: строка, чтобы Pydantic не ругался
                transdatetime=str(getattr(row, "transdatetime")),
                # если TransactionInput описывает id/target/behavioral_patterns — можно дополнительно мапнуть
            )
            transactions.append(trans)

        # ---- Предсказание пачкой ----
        outputs: List[TransactionOutput] = await run_in_threadpool(
            fraud_detector.predict_batch,
            transactions,
        )

        # ---- Добавляем результаты в DataFrame ----
        df["fraud_score"] = [o.fraud_probability for o in outputs]
        df["prediction"] = [1 if o.is_fraud else 0 for o in outputs]
        df["risk_level"] = [o.risk_level for o in outputs]
        df["model_version"] = [o.model_version for o in outputs]
        df["threshold_used"] = [o.threshold_used for o in outputs]
        df["score_catboost"] = [
            o.individual_scores.catboost for o in outputs
        ]
        df["score_xgboost"] = [
            o.individual_scores.xgboost for o in outputs
        ]
        df["score_lightgbm"] = [
            o.individual_scores.lightgbm for o in outputs
        ]
        df["anomaly_score"] = [
            o.individual_scores.anomaly for o in outputs
        ]

        # ---- Конвертация обратно в CSV ----
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_result = csv_buffer.getvalue()

        return Response(
            content=csv_result,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=result_with_scores.csv"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=Stats)
async def get_stats():
    """Get model statistics (features, threshold, history size)."""
    try:
        result = await run_in_threadpool(fraud_detector.get_stats)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health_check():
    """Health check для модели."""
    return {
        "status": "ok",
        "stats": fraud_detector.get_stats().model_dump(),
    }
    