FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt ./requirements.txt
COPY service/requirements-service.txt ./requirements-service.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt -r requirements-service.txt
COPY src ./src
COPY data ./data
COPY service/app ./app
ENV RESOLVR_SRC=/srv/src DATA_DIR=/srv/data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
