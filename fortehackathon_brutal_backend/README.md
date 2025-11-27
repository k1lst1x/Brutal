Fortebank AI hackathon Brutal project

Система для детекции мошеннических транзакций в реальном времени
Использует ансамбль моделей CatBoost + XGBoost + LightGBM + IsolationForest + динамическую историю клиента.

🚀 Возможности

🔥 Реальное время: обработка транзакций ~1–3 ms

🤖 Ансамбль моделей: CatBoost, XGBoost, LightGBM

🧭 Аномалист: Isolation Forest для anomaly score

🧩 60+ признаков: временные, графовые, динамические, статистические, поведенческие

📈 Авто-обновление истории: пользовательская активность хранится 60 дней

📦 Batch API: массовая обработка транзакций

🏷 Risk scoring: LOW / MEDIUM / HIGH / CRITICAL

⚠️ Человеко-читаемые алерты

🔧 Установка
git clone https://github.com/your-repo/fraud-api
cd fortehackathon_brutal_backend
pip install -r requirements.txt


Убедитесь, что Python ≥ 3.10.

🧠 Модель (model_package.pkl)

Пример JSON транзакции

transaction = {
    "cst_dim_id": 1234,
    "transdatetime": "2024-01-15 23:45:00",
    "amount": 50000,
    "direction": "card_transfer"
}

⚡ FastAPI REST API
▶ Запуск
uvicorn fortehackathon_brutal_backend.app:app --host 0.0.0.0 --port 8000

▶ POST /predict

Request

{
  "cst_dim_id": 1234,
  "transdatetime": "2024-01-15 23:45:00",
  "amount": 50000,
  "direction": "card_transfer"
}


Response

{
  "is_fraud": true,
  "fraud_probability": 0.93,
  "risk_level": "CRITICAL",
  "alerts": [
    "⚠️ Amount is 3x higher than 30-day average",
    "⚠️ Transaction during night hours (23:00-06:00)"
  ],
  "processing_time_ms": 1.57,
  "individual_scores": {
    "catboost": 0.91,
    "xgboost": 0.89,
    "lightgbm": 0.87,
    "anomaly": 0.12
  }
}

📚 Возможные алерты
Алерт	Значение
⚠️ Amount is 3x higher than 30-day average	Аномальный размер
⚠️ Transaction less than 1 hour since last one	Слишком частые операции
⚠️ Transaction during night hours	Нетипичное время
⚠️ Sudden increase in velocity	Разгон по количеству
⚠️ New customer with limited history	Молодой клиент
🚨 CRITICAL	Чрезвычайно высокий риск
🧮 Фичи (основные группы)

Динамические: за 7 и 30 дней

Velocity / acceleration

Статистика суммы: std, max, ratios

Временные: час, месяц, день недели

Аномалии: time_since_last, amount spike

Графовые: out-degree, in-degree

Поведенческие паттерны: опционально

🐳 Docker 
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]


Запуск:

cd fortehackathon_brutal_backend

docker build -t fortehackathon_brutal_backend .
docker run -p 8000:8000 fortehackathon_brutal_backend

Нужен асинхронный PostgreSQL 

Пример url: postgresql+asyncpg://{USER}:{PASS}@{HOST}:{DB_PORT}/{DB_NAME}

📜 Лицензия

MIT License.
