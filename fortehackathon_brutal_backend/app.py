from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from router import router as api_router  # router is defined in router.py

app = FastAPI(title="Brutal Fraud Shield API")

# --- CORS НАСТРОЙКА ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # можно ["*"] на демо, но лучше так
    allow_credentials=True,
    allow_methods=["*"],          # разрешаем все методы (включая OPTIONS)
    allow_headers=["*"],
)

# --- Подключаем твой router с /predict, /bulk_predict и т.д. ---
# Router is already imported as `api_router` above
app.include_router(api_router)

@app.get("/")
def root():
    return {"status": "ok", "message": "Brutal Fraud Shield API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, lifespan="on", reload=True)
