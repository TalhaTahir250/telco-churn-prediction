FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5

COPY requirements.txt .

RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir pandas==2.2.2
RUN pip install --no-cache-dir scikit-learn==1.7.1
RUN pip install --no-cache-dir xgboost==2.0.3
RUN pip install --no-cache-dir fastapi==0.111.0 "uvicorn[standard]==0.30.1" pydantic==2.7.1 joblib==1.4.2

COPY src/ ./src/
COPY model.pkl .

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]