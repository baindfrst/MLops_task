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
4 | Собрать FAISS‑индекс |
5 | Подключить reranker (пока не сделал) |
6 | Дообучить генератор на SberQuAD Q/A (пока что использую открытую Qwen 2) |
7 | Проверка метрик (пока что только метрики retriever) |
8 | CLI / web‑демо |
9 | Сделать тесты |

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
### Запуск локального API
```
uvicorn src.api.rag_app_local:app --reload --port 8002
```


---