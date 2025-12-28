FROM python:3.11

WORKDIR /app

COPY api/requirements.txt requirements_api.txt
COPY ui/requirements.txt requirements_ui.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_api.txt && \
    pip install --no-cache-dir -r requirements_ui.txt 

COPY api ./api
COPY ui ./ui

CMD uvicorn api.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run ui/app.py --server.port 7860 --server.address 0.0.0.0