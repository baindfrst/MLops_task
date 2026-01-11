## Цель проекта

Создать RAG‑бота, который сможет:

- понимать вопросы на русском
- искать ответы в русских текстах (SberQuAD)
- работать офлайн на домашнем ПК

---


## Архитектура

* Запрос пользователя
* Модель‑ретривер
* Реранкер
* Генеративная модель
* Ответ

---

## Данные

| Название | HuggingFace |
|---|---|
SberQuAD | `sberbank-ai/sberquad` |
---

## Метрики

### Качество
| Метрика | Цель |
|---|---|
Recall@10 | ≥ 0.90  
ROUGE‑L | ≥ 0.40  
BLEU | ≥ 0.25  

### Производительность
| Метрика | Цель |
---|---|
End‑to‑end latency | ≤ 1.2 c  
GPU VRAM | ≤ 7.5 GB  

---


## План экспериментов

| Этап | Цель |
|---|---|
1 | Загрузить SberQuAD |
2 | Подготовить данные |
3 | Обучить retriever |
4 | Собрать FAISS‑индекс (пока что просто эмбендинги через pickle) |
5 | Подключить reranker (пока не сделал) |
6 | Дообучить генератор на SberQuAD Q/A (пока что использую открытую Qwen 2) |
7 | Проверка метрик (пока что только метрики retriever) |
8 | CLI / web‑демо |
9 | Сделать тесты |
10 | Добавить DVC |

---

##  Быстрый старт

### Установка зависимостей
```
pip install -r requirements.txt (или requirements_gpu.txt если есть GPU)
```

### Загрузка данных

Добавить свой HF_TOKEN в окружение  

```
python -m src.data.load_sberquad
```

### Подготовка данных
```
python -m src.data.preprocess --config configs/data.yaml
```

### Обучение ретривера
```
python -m src.retriever.train --config configs/train_retriever.yaml
```

### Валидация
```
python -m src.retriever.eval --model artifacts/retriever_ru
```

### Тестирование
```
pytest -q
```

### Построение индексов
```
python -m src.retriever.build_index
```

### Запуск локального API
```
uvicorn src.api.rag_app:app --host 0.0.0.0 --port 8000 --reload

перейти на http://127.0.0.1:8000/docs
```

## DVC
В качестве места для хранения данных для dvc используется docker compose с minio. Его можно развернуть как локально, так и вытянуть данные с VPS, который я отжал на время у друга
Под контролем DVC находятся:

* обработанные датасеты:
  * `data/processed/*.jsonl`
* артефакты обучения:
  * `artifacts/retriever_ru/`
  * `artifacts/passages_index.pkl`
* итоговая модель для инференса:
  * `model-store/model.pt`

Если хотите запустить docker c minio:  
```
docker compose up -d  
```

Web UI: http://localhost:9001

S3 endpoint: http://localhost:9000

для того что бы получить данные нужно задать 2 переменных окружения  
```
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123
```
Для подключени к VPS где уже лежат данные стоит использовать следующий remote (он уже прописан в config):
```
dvc remote add -d minio s3://dvcstor/art
dvc remote modify minio endpointurl http://94.183.187.106:9000
dvc remote modify minio use_ssl false
```

Для загрузки данных:  
```
git clone <repo>
pip install -r requirements.txt
dvc pull
dvc repro
```

## MLlfow

mlflow используется в src.retriever.train те при запуске обучения  
```
python -m src.retriever.train --config configs/train_retriever.yaml
```
логируются:
* гиперпараметры модели
* параметры обучения
* пути к данным
* значения метрик качества на train / validation
* итоговые показатели модели
* обученная модель

Артефакты mlflow так же находятся под dvc

для запуска UI:

```
mlflow ui
```
и перейти в http://127.0.0.1:5000

## Docker для инферинса:
собрать докер
```
docker build -t app:v1 .
```
запуск предсказания:
```
docker run --rm -v "${PWD}\inputs:/app/inputs" -v "${PWD}\outputs:/app/outputs" app:v1 --input_path /app/inputs/sample_queries.csv --output_path /app/outputs/preds.csv
```

где \inputs и \outputs локальные папки которые матируются в /app/inputs и /app/outputs и в них задаются файлы для входа и выхода

к примеру в команде выше в локальной папке *inputs* лежит файл *sample_queries.csv*, а выход будет в папке *outputs* в файле *preds.csv*

вход (.csv файл):
```
query
что такое
как открыть
```
выход будет csv файл с колонами **query,top_pids,top_scores,top_texts**

## Docker для torchserver

Собрать образ TorchServe:
```
docker build -f Dockerfile.torchserve -t mymodel-serve:v1 .
```

для сборки нужны model-store/model.pt, а также artifacts/ и data/processed/

запуск сервера:

```
docker run --rm -p 8080:8080 -p 8081:8081 mymodel-serve:v1
```

Инференс делается POST запросом на http://127.0.0.1:8080/predictions/mymodel

ожидает JSON вида:
```
{"query": "что такое dvc", "k": 3}
```
query — строка запроса
k — top-k

пример запроса:
```
curl -X POST http://localhost:8080/predictions/mymodel -T sample_input.json
```

---