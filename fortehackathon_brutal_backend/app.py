from fastapi import FastAPI
from fortehackathon_brutal_backend.router import router

app = FastAPI(title="Fraud Detection API")

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, lifespan="on", reload=True)
