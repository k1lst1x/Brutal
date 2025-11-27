"""
API для реал-тайм детекции мошеннических транзакций
Использует обученную модель из fraud_model_production/
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

from config import MODEL_DIR
from dtos import TransactionOutput, TransactionInput, Stats, Models


class FraudDetectionAPI:
    """
    API для детекции фрода
    """

    def __init__(self, model_path: str = MODEL_DIR):
        """
        Загружает обученную модель
        """
        print(f"Loading model from {model_path}...")
        self.model_pkg = joblib.load(model_path)

        self.iso = self.model_pkg['iso']
        self.catboost = self.model_pkg['catboost']
        self.xgboost = self.model_pkg['xgboost']
        self.lightgbm = self.model_pkg['lightgbm']
        self.threshold = self.model_pkg['threshold']
        self.feature_cols = self.model_pkg['feature_cols']
        self.encoders = self.model_pkg['encoders']
        self.weights = self.model_pkg['ensemble_weights']
        self.history = self.model_pkg.get('history', {})

        print(f"✓ Model loaded successfully")
        print(f"  Version: {self.model_pkg.get('version', 'unknown')}")
        print(f"  Threshold: {self.threshold:.4f}")
        print(f"  Features: {len(self.feature_cols)}")

    def predict_single_transaction(
            self,
            transaction: TransactionInput,
            behavioral_patterns: Dict[str, Any] = None
    ) -> TransactionOutput:
        """
        Предсказывает вероятность фрода для одной транзакции

        Args:
            transaction: словарь с данными транзакции
                Required: cst_dim_id, transdatetime, amount, direction
            behavioral_patterns: словарь с поведенческими паттернами клиента (опционально)

        Returns:
            TransactionOutput (
                is_fraud: bool
                fraud_probability: float
                risk_level: str
                alerts: list
                processing_time_ms: float
            )
        """
        start_time = datetime.now()
        transaction_dict = transaction.model_dump()

        # Валидация входных данных
        required_fields = ['cst_dim_id', 'transdatetime', 'amount', 'direction']
        for field in required_fields:
            if field not in transaction_dict:
                raise ValueError(f"Missing required field: {field}")

        # Построение фичей
        features = self._build_features(transaction, behavioral_patterns)

        # Anomaly score
        X_single = pd.DataFrame([features])[self.feature_cols]
        X_single = X_single.apply(pd.to_numeric, errors='coerce').fillna(0)

        anomaly_score = -self.iso.decision_function(X_single)[0]
        X_single['anomaly_score'] = anomaly_score

        # Ансамбль предсказаний
        p_cat = self.catboost.predict_proba(X_single)[0, 1]
        p_xgb = self.xgboost.predict_proba(X_single)[0, 1]
        p_lgb = self.lightgbm.predict_proba(X_single)[0, 1]

        fraud_prob = (
                self.weights[0] * p_cat +
                self.weights[1] * p_xgb +
                self.weights[2] * p_lgb
        )

        is_fraud = fraud_prob >= self.threshold

        # Определяем уровень риска
        if fraud_prob >= 0.8:
            risk_level = "CRITICAL"
        elif fraud_prob >= 0.6:
            risk_level = "HIGH"
        elif fraud_prob >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Генерируем алерты
        alerts = self._generate_alerts(transaction, features, fraud_prob)

        # Обновляем историю (для следующих транзакций)
        self._update_history(transaction)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return TransactionOutput(
            is_fraud=bool(is_fraud),
            fraud_probability=float(fraud_prob),
            risk_level=risk_level,
            alerts=alerts,
            processing_time_ms=processing_time,
            model_version=self.model_pkg.get('version', 'unknown'),
            threshold_used=self.threshold,
            individual_scores=Models(
                catboost=float(p_cat),
                xgboost=float(p_xgb),
                lightgbm=float(p_lgb),
                anomaly=float(anomaly_score),
            )
        )

    def predict_batch(
            self,
            transactions: List[TransactionInput],
            behavioral_patterns: Dict[int, Dict[str, Any]] = None
    ) -> List[TransactionOutput]:
        """
        Предсказание для нескольких транзакций
        """
        results = []
        for trans in transactions:
            cst_id = trans.cst_dim_id
            patterns = behavioral_patterns.get(cst_id) if behavioral_patterns else None
            result = self.predict_single_transaction(trans, patterns)
            results.append(result)

        return results

    def _build_features(
            self,
            transaction: TransactionInput,
            behavioral_patterns: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Строит фичи для одной транзакции
        """
        cst_id = transaction.cst_dim_id
        ts = pd.to_datetime(transaction.transdatetime)
        amount = float(transaction.amount)
        direction = str(transaction.direction)

        # История клиента
        hist = self.history.get(cst_id, [])

        cutoff_7 = ts - timedelta(days=7)
        cutoff_30 = ts - timedelta(days=30)
        recent_7 = [h for h in hist if h[0] >= cutoff_7 and h[0] < ts]
        recent_30 = [h for h in hist if h[0] >= cutoff_30 and h[0] < ts]

        features = {}

        # Базовые динамические
        features['num_trans_last_7d'] = len(recent_7)
        features['num_trans_last_30d'] = len(recent_30)
        features['sum_amount_last_7d'] = sum(h[1] for h in recent_7)
        features['sum_amount_last_30d'] = sum(h[1] for h in recent_30)

        avg_7 = features['sum_amount_last_7d'] / features['num_trans_last_7d'] if features[
                                                                                      'num_trans_last_7d'] > 0 else 0
        avg_30 = features['sum_amount_last_30d'] / features['num_trans_last_30d'] if features[
                                                                                         'num_trans_last_30d'] > 0 else 0

        features['avg_amount_last_7d'] = avg_7
        features['avg_amount_last_30d'] = avg_30

        # Velocity
        features['velocity_7d'] = features['num_trans_last_7d'] / 7.0
        features['velocity_30d'] = features['num_trans_last_30d'] / 30.0
        features['amount_velocity_7d'] = features['sum_amount_last_7d'] / 7.0
        features['amount_velocity_30d'] = features['sum_amount_last_30d'] / 30.0
        features['velocity_acceleration'] = features['velocity_7d'] - features['velocity_30d']

        # Распределение
        amounts_7d = [h[1] for h in recent_7]
        features['std_amount_7d'] = np.std(amounts_7d) if len(amounts_7d) > 1 else 0
        features['max_amount_7d'] = max(amounts_7d) if amounts_7d else 0
        features['min_amount_7d'] = min(amounts_7d) if amounts_7d else 0

        # Ratios
        features['ratio_num_7_30'] = features['num_trans_last_7d'] / features['num_trans_last_30d'] if features[
                                                                                                           'num_trans_last_30d'] > 0 else 0
        features['ratio_sum_7_30'] = features['sum_amount_last_7d'] / features['sum_amount_last_30d'] if features[
                                                                                                             'sum_amount_last_30d'] > 0 else 0
        features['amount_ratio_avg7'] = amount / avg_7 if avg_7 > 0 else 0
        features['amount_ratio_avg30'] = amount / avg_30 if avg_30 > 0 else 0
        features['amount_to_max_ratio'] = amount / features['max_amount_7d'] if features['max_amount_7d'] > 0 else 0

        # Временные
        last_ts = hist[-1][0] if hist else None
        features['time_since_last_hours'] = (ts - last_ts).total_seconds() / 3600.0 if last_ts else 0
        features['time_since_last_squared'] = features['time_since_last_hours'] ** 2

        first_ts = hist[0][0] if hist else ts
        features['days_since_first'] = (ts - first_ts).days
        features['trans_frequency'] = len(hist) / features['days_since_first'] if features[
                                                                                      'days_since_first'] > 0 else 0

        # Графовые
        features['num_prev_trans_to_same'] = sum(1 for h in hist if h[2] == direction and h[0] < ts)
        features['total_prev_trans'] = len([h for h in hist if h[0] < ts])
        features['unique_directions_count'] = len(set(h[2] for h in hist if h[0] < ts))

        # Граф (упрощенно, без полной статистики)
        features['sender_out_degree'] = features['unique_directions_count']
        features['receiver_in_degree'] = 1  # Не можем посчитать без других клиентов
        features['pair_count'] = features['num_prev_trans_to_same']

        # Аномалии
        features['is_amount_spike'] = 1 if (avg_30 > 0 and amount > avg_30 * 3) else 0
        features['is_rapid_repeat'] = 1 if (
                features['time_since_last_hours'] < 1.0 and features['time_since_last_hours'] > 0) else 0

        hour = ts.hour
        features['is_night_transaction'] = 1 if (hour >= 23 or hour <= 6) else 0
        features['is_weekend'] = 1 if ts.dayofweek in [5, 6] else 0

        # Временные фичи
        features['hour'] = hour
        features['dayofweek'] = ts.dayofweek
        features['month'] = ts.month
        features['amount'] = amount
        features['amount_log'] = np.log1p(amount)

        # Энкодинг direction
        if 'direction' in self.encoders:
            le = self.encoders['direction']
            if direction in le.classes_:
                features['direction'] = le.transform([direction])[0]
            else:
                features['direction'] = le.transform([le.classes_[0]])[0]
        else:
            features['direction'] = 0

        # Поведенческие паттерны (если есть)
        if behavioral_patterns:
            for key, value in behavioral_patterns.items():
                if key not in ['cst_dim_id', 'transdate']:
                    features[key] = value
        else:
            # Заполняем дефолтными значениями
            for col in self.feature_cols:
                if col not in features:
                    features[col] = 0

        return features

    def _generate_alerts(
            self,
            transaction: TransactionInput,  # not actual
            features: Dict[str, float],
            fraud_prob: float
    ) -> List[str]:
        """
        Генерирует человеко-читаемые алерты
        """
        alerts = []

        if features.get('is_amount_spike', 0) == 1:
            alerts.append("⚠️ Amount is 3x higher than 30-day average")

        if features.get('is_rapid_repeat', 0) == 1:
            alerts.append("⚠️ Transaction less than 1 hour since last one")

        if features.get('is_night_transaction', 0) == 1:
            alerts.append("⚠️ Transaction during night hours (23:00-06:00)")

        if features.get('velocity_acceleration', 0) > 2:
            alerts.append("⚠️ Sudden increase in transaction velocity")

        if features.get('total_prev_trans', 0) < 5:
            alerts.append("⚠️ New customer with limited history")

        if fraud_prob > 0.9:
            alerts.append("🚨 CRITICAL: Very high fraud probability")

        return alerts

    def _update_history(self, transaction: TransactionInput):
        """
        Обновляет историю транзакций клиента
        """
        cst_id = transaction.cst_dim_id
        ts = pd.to_datetime(transaction.transdatetime)
        amount = float(transaction.amount)
        direction = str(transaction.direction)

        if cst_id not in self.history:
            self.history[cst_id] = []

        self.history[cst_id].append((ts, amount, direction))

        # Храним только последние 60 дней
        cutoff = ts - timedelta(days=60)
        self.history[cst_id] = [h for h in self.history[cst_id] if h[0] >= cutoff]

    def get_stats(self) -> Stats:
        """
        Возвращает статистику API
        """
        return Stats(
            total_customers_in_history=len(self.history),
            model_version=self.model_pkg.get('version', 'unknown'),
            threshold=self.threshold,
            num_features=len(self.feature_cols),
        )
