FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY app.py main.py ./

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
