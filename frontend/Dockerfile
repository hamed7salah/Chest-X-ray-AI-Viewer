FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY frontend /app
EXPOSE 8501
CMD ["gunicorn", "--bind", "0.0.0.0:8501", "app:app", "--workers", "1", "--threads", "4"]
