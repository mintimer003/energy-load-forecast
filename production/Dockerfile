# Один образ на все роли продакшн-слоя: train / predict / serve / monitor.
# Роль выбирается командой при запуске контейнера (см. docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app

# зависимости отдельным слоем — кэш пересборки
COPY production/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY production/*.py ./

# данные и артефакты монтируются томами:
#   /data    — CSV с историей (только чтение)
#   /models  — обученные модели + manifest
#   /preds   — прогнозы + status.json
ENV DATA_DIR=/data MODELS_DIR=/models PREDICTIONS_DIR=/preds

CMD ["python", "train.py", "--data-dir", "/data", "--models-dir", "/models"]
