FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /fortehackathon_brutal_backend

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "fortehackathon_brutal_backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
