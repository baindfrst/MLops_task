FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY configs /app/configs

COPY dvc.yaml dvc.lock /app/
COPY .dvc /app/.dvc

COPY artifacts /app/artifacts
COPY data/processed /app/data/processed

ENTRYPOINT ["python", "-m", "src.predict"]
